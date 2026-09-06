//! 在容器的 X11 会话中执行输入；连接只释放自己按下的键和按钮。
use base64::{Engine, engine::general_purpose::STANDARD};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    error::Error,
    ffi::CString,
    io::{self, BufRead, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};
use x11rb::{
    connection::Connection,
    protocol::{
        xfixes::ConnectionExt as _,
        xproto::{self, ConnectionExt as _},
        xtest::ConnectionExt as _,
    },
    rust_connection::RustConnection,
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
#[derive(Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
struct Point {
    x: i16,
    y: i16,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    id: u64,
    method: String,
    input: Value,
}
#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Input {
    x: Option<i16>,
    y: Option<i16>,
    key: Option<String>,
    text: Option<String>,
    mouse_button: Option<String>,
    click_count: Option<u16>,
    duration: Option<u64>,
    direction: Option<String>,
    pixels: Option<u32>,
    path: Option<Vec<Point>>,
    action: Option<String>,
}
struct Desktop {
    connection: RustConnection,
    root: u32,
    screen: usize,
    keys: Vec<u8>,
    buttons: Vec<u8>,
    cancelled: Arc<AtomicBool>,
    clipboard: x11_clipboard::Clipboard,
}
impl Desktop {
    /// 建立本进程独占的输入状态；不改变其他客户端已经按住的键。
    fn connect(cancelled: Arc<AtomicBool>) -> Result<Self> {
        let (connection, screen) = x11rb::connect(None)?;
        let root = connection.setup().roots[screen].root;
        connection.xtest_get_version(2, 2)?.reply()?;
        connection.xfixes_query_version(5, 0)?.reply()?;
        Ok(Self {
            connection,
            root,
            screen,
            keys: vec![],
            buttons: vec![],
            cancelled,
            clipboard: x11_clipboard::Clipboard::new()?,
        })
    }
    fn check(&self) -> Result<()> {
        if self.cancelled.load(Ordering::Relaxed) {
            return Err("operation cancelled; prior effects may remain".into());
        }
        Ok(())
    }
    /// 等待过程中响应取消，保证 RAII 能够执行释放。
    fn wait(&self, ms: u64) -> Result<()> {
        let until = Instant::now() + Duration::from_millis(ms);
        while Instant::now() < until {
            self.check()?;
            thread::sleep(
                until
                    .saturating_duration_since(Instant::now())
                    .min(Duration::from_millis(10)),
            );
        }
        self.check()
    }
    fn event(&self, kind: u8, detail: u8, point: Point) -> Result<()> {
        self.connection
            .xtest_fake_input(kind, detail, 0, self.root, point.x, point.y, 0)?
            .check()?;
        self.connection.flush()?;
        Ok(())
    }
    fn position(&self) -> Result<Point> {
        let cursor = self.connection.query_pointer(self.root)?.reply()?;
        Ok(Point {
            x: cursor.root_x,
            y: cursor.root_y,
        })
    }
    fn move_to(&self, point: Point) -> Result<()> {
        self.check()?;
        let geometry = self.connection.get_geometry(self.root)?.reply()?;
        if point.x < 0
            || point.y < 0
            || point.x >= geometry.width as i16
            || point.y >= geometry.height as i16
        {
            return Err("point is outside the desktop".into());
        }
        let start = self.position()?;
        if start.x == point.x && start.y == point.y {
            return self.event(xproto::MOTION_NOTIFY_EVENT, 0, point);
        }
        // 原 move_path@0x56fb0：ceil(distance / 160 * 10)，夹到 1..10，间隔 8ms。
        let dx = f64::from(point.x) - f64::from(start.x);
        let dy = f64::from(point.y) - f64::from(start.y);
        let steps = (dx.hypot(dy) / 160.0 * 10.0).ceil().clamp(1.0, 10.0) as u32;
        let mut previous = None;
        for step in 0..=steps {
            let ratio = f64::from(step) / f64::from(steps);
            let current = Point {
                x: (f64::from(start.x) + dx * ratio).round() as i16,
                y: (f64::from(start.y) + dy * ratio).round() as i16,
            };
            if previous == Some((current.x, current.y)) {
                continue;
            }
            self.event(xproto::MOTION_NOTIFY_EVENT, 0, current)?;
            previous = Some((current.x, current.y));
            if step < steps {
                self.wait(8)?;
            }
        }
        Ok(())
    }
    /// 查询当前键盘映射，而不是假定 Xvnc 使用固定 keycode。
    fn keycodes(&self, chord: &str) -> Result<Vec<u8>> {
        let xlib = x11_dl::xlib::Xlib::open()?;
        let setup = self.connection.setup();
        let mapping = self
            .connection
            .get_keyboard_mapping(setup.min_keycode, setup.max_keycode - setup.min_keycode + 1)?
            .reply()?;
        let mut result = vec![];
        for part in chord.split('+') {
            let name = match part.to_ascii_lowercase().as_str() {
                "ctrl" | "control" => "Control_L",
                "alt" => "Alt_L",
                "shift" => "Shift_L",
                "super" | "meta" | "cmd" | "command" => "Super_L",
                "enter" => "Return",
                "esc" => "Escape",
                "backspace" => "BackSpace",
                "delete" => "Delete",
                "space" => "space",
                _ => part,
            };
            let name = CString::new(name)?;
            // XStringToKeysym 只读取以 NUL 结尾的名称，不持有传入指针。
            let symbol = unsafe { (xlib.XStringToKeysym)(name.as_ptr()) } as u32;
            if symbol == 0 {
                return Err(format!("unknown key: {part}").into());
            }
            let found = mapping
                .keysyms
                .chunks(mapping.keysyms_per_keycode as usize)
                .enumerate()
                .find_map(|(offset, symbols)| {
                    symbols
                        .iter()
                        .position(|value| *value == symbol)
                        .map(|column| (setup.min_keycode + offset as u8, column))
                });
            let (code, column) = found.ok_or_else(|| format!("key is not mapped: {part}"))?;
            if column % 2 == 1 {
                let shift = self.keycodes("Shift_L")?[0];
                if !result.contains(&shift) {
                    result.push(shift);
                }
            }
            if !result.contains(&code) {
                result.push(code);
            }
        }
        Ok(result)
    }
    fn press_keys(&mut self, chord: Option<&str>) -> Result<()> {
        let Some(chord) = chord else {
            return Ok(());
        };
        let codes = self.keycodes(chord)?;
        let down = self.connection.query_keymap()?.reply()?.keys;
        for code in codes {
            self.check()?;
            if down[code as usize / 8] & (1 << (code % 8)) != 0 || self.keys.contains(&code) {
                continue;
            }
            self.keys.push(code);
            self.event(xproto::KEY_PRESS_EVENT, code, self.position()?)?;
        }
        Ok(())
    }
    fn press_button(&mut self, button: u8) -> Result<()> {
        self.check()?;
        let current = self.connection.query_pointer(self.root)?.reply()?;
        if button <= 5 && u16::from(current.mask) & (1 << (7 + button)) != 0 {
            return Err("mouse button is already held by another input owner".into());
        }
        self.buttons.push(button);
        self.event(xproto::BUTTON_PRESS_EVENT, button, self.position()?)
    }
    fn release_button(&mut self, button: u8) -> Result<()> {
        self.event(xproto::BUTTON_RELEASE_EVENT, button, self.position()?)?;
        self.buttons.retain(|owned| *owned != button);
        Ok(())
    }
    /// 释放失败必须报告；只有成功发送 release 后才移除 owned 记录。
    fn release(&mut self) -> Result<()> {
        while let Some(button) = self.buttons.last().copied() {
            self.release_button(button)?;
        }
        while let Some(code) = self.keys.last().copied() {
            self.event(xproto::KEY_RELEASE_EVENT, code, self.position()?)?;
            self.keys.pop();
        }
        Ok(())
    }
    /// 对相同起终点直接发送一次 motion，避免等待永远不会发生的位移。
    fn move_path(&self, path: &[Point]) -> Result<()> {
        for point in path {
            self.move_to(*point)?;
        }
        Ok(())
    }
    /// 临时替换纯文本剪贴板，粘贴完成后恢复原文本；不保存截图或剪贴板文件。
    fn type_text(&mut self, text: &str) -> Result<()> {
        let atoms = &self.clipboard.getter.atoms;
        let selection = atoms.clipboard;
        let target = atoms.utf8_string;
        let owner = self
            .connection
            .get_selection_owner(selection)?
            .reply()?
            .owner;
        let before = if owner == 0 {
            None
        } else {
            Some(
                self.clipboard
                    .load(selection, target, atoms.property, Duration::from_secs(2))?,
            )
        };
        self.clipboard
            .store(selection, target, text.as_bytes().to_vec())?;
        let result = (|| {
            self.press_keys(Some("ctrl+v"))?;
            self.wait(30)?;
            self.release()?;
            self.wait(100)
        })();
        let restore: Result<()> = match before {
            Some(bytes) => self
                .clipboard
                .store(selection, target, bytes)
                .map_err(Into::into),
            None => self
                .connection
                .set_selection_owner(0u32, selection, x11rb::CURRENT_TIME)?
                .check()
                .map_err(Into::into),
        };
        restore?;
        result
    }
    /// 从根窗口读取像素，并叠加 XFixes 返回的真实鼠标指针。
    fn screenshot(&self) -> Result<Value> {
        let geometry = self.connection.get_geometry(self.root)?.reply()?;
        let reply = self
            .connection
            .get_image(
                xproto::ImageFormat::Z_PIXMAP,
                self.root,
                0,
                0,
                geometry.width,
                geometry.height,
                u32::MAX,
            )?
            .reply()?;
        let setup = self.connection.setup();
        let format = setup
            .pixmap_formats
            .iter()
            .find(|format| format.depth == reply.depth)
            .ok_or("unknown pixel format")?;
        let visual = setup.roots[self.screen]
            .allowed_depths
            .iter()
            .flat_map(|depth| &depth.visuals)
            .find(|visual| visual.visual_id == reply.visual)
            .ok_or("unknown visual")?;
        if ![16, 24, 32].contains(&format.bits_per_pixel) {
            return Err("unsupported screenshot pixel depth".into());
        }
        let bytes = format.bits_per_pixel as usize / 8;
        let row_bits = geometry.width as usize * format.bits_per_pixel as usize;
        let stride =
            row_bits.div_ceil(format.scanline_pad as usize) * format.scanline_pad as usize / 8;
        let channel = |pixel: u32, mask: u32| -> u8 {
            (((pixel & mask) >> mask.trailing_zeros()) * 255 / (mask >> mask.trailing_zeros()))
                as u8
        };
        let mut image = image::RgbImage::new(geometry.width.into(), geometry.height.into());
        for (x, y, rgb) in image.enumerate_pixels_mut() {
            let data = &reply.data[y as usize * stride + x as usize * bytes..][..bytes];
            let mut pixel = 0u32;
            for (i, byte) in data.iter().enumerate() {
                let shift = if setup.image_byte_order == xproto::ImageOrder::LSB_FIRST {
                    i
                } else {
                    bytes - i - 1
                };
                pixel |= u32::from(*byte) << (shift * 8);
            }
            *rgb = image::Rgb([
                channel(pixel, visual.red_mask),
                channel(pixel, visual.green_mask),
                channel(pixel, visual.blue_mask),
            ]);
        }
        let cursor = self.connection.xfixes_get_cursor_image()?.reply()?;
        for cy in 0..cursor.height {
            for cx in 0..cursor.width {
                let x = i32::from(cursor.x) - i32::from(cursor.xhot) + i32::from(cx);
                let y = i32::from(cursor.y) - i32::from(cursor.yhot) + i32::from(cy);
                if x < 0
                    || y < 0
                    || x >= i32::from(geometry.width)
                    || y >= i32::from(geometry.height)
                {
                    continue;
                }
                let pixel = cursor.cursor_image[cy as usize * cursor.width as usize + cx as usize];
                let alpha = pixel >> 24;
                let rgb = image.get_pixel_mut(x as u32, y as u32);
                for (i, shift) in [16, 8, 0].iter().enumerate() {
                    // XFixes 返回预乘 alpha 的 ARGB。
                    rgb[i] = (((pixel >> shift) & 255) + u32::from(rgb[i]) * (255 - alpha) / 255)
                        .min(255) as u8;
                }
            }
        }
        let mut encoded = vec![];
        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut encoded, 80)
            .encode_image(&image)?;
        Ok(
            json!({"mimeType":"image/jpeg", "data":STANDARD.encode(encoded), "width":geometry.width, "height":geometry.height}),
        )
    }
    /// 在一个串行输入连接上执行公开命令；drag_handle 的状态由 JS 调用持有。
    fn execute(&mut self, method: &str, input: Input) -> Result<Value> {
        self.check()?;
        if input.duration.is_some_and(|value| value > 30_000) {
            return Err("duration exceeds 30000 ms".into());
        }
        let point = || -> Result<Point> {
            Ok(Point {
                x: input.x.ok_or("x is required")?,
                y: input.y.ok_or("y is required")?,
            })
        };
        match method {
            "get_screenshot" => return self.screenshot(),
            "release" => self.release()?,
            "move" => {
                self.press_keys(input.key.as_deref())?;
                self.move_to(point()?)?;
                self.release()?;
            }
            "click" => {
                let button = match input.mouse_button.as_deref().unwrap_or("left") {
                    "left" | "l" => 1,
                    "middle" | "m" => 2,
                    "right" | "r" => 3,
                    _ => return Err("invalid mouse_button".into()),
                };
                let count = input.click_count.unwrap_or(1);
                if count == 0 || count > 10 {
                    return Err("click_count must be 1..10".into());
                }
                self.move_to(point()?)?;
                self.press_keys(input.key.as_deref())?;
                for _ in 0..count {
                    self.press_button(button)?;
                    self.wait(input.duration.unwrap_or(0))?;
                    self.release_button(button)?;
                    self.wait(50)?;
                }
                self.release()?;
            }
            "drag" => {
                let path = input.path.ok_or("path is required")?;
                if path.len() < 2 || path.len() > 4096 {
                    return Err("path must contain 2..4096 points".into());
                }
                self.move_to(path[0])?;
                self.press_keys(input.key.as_deref())?;
                self.press_button(1)?;
                self.move_path(&path[1..])?;
                self.release()?;
            }
            "drag_handle" => match input.action.as_deref() {
                Some("start") => {
                    if !self.buttons.is_empty() {
                        return Err("a drag is already active".into());
                    }
                    self.move_to(point()?)?;
                    self.press_keys(input.key.as_deref())?;
                    self.press_button(1)?;
                }
                Some("move_to") => {
                    if !self.buttons.contains(&1) {
                        return Err("drag is not active".into());
                    }
                    self.move_to(point()?)?;
                }
                Some("end") => {
                    if !self.buttons.contains(&1) {
                        return Err("drag is not active".into());
                    }
                    self.release()?;
                }
                _ => return Err("invalid drag_handle action".into()),
            },
            "press_key" => {
                self.press_keys(Some(input.key.as_deref().ok_or("key is required")?))?;
                self.wait(input.duration.unwrap_or(0))?;
                self.release()?;
            }
            "type_text" => self.type_text(input.text.as_deref().ok_or("text is required")?)?,
            "scroll" => {
                let button = match input.direction.as_deref() {
                    Some("up" | "u") => 4,
                    Some("down" | "d") => 5,
                    Some("left" | "l") => 6,
                    Some("right" | "r") => 7,
                    _ => return Err("invalid scroll direction".into()),
                };
                let pixels = input.pixels.unwrap_or(500);
                if pixels > 100_000 {
                    return Err("pixels exceeds 100000".into());
                }
                if input.x.is_some() || input.y.is_some() {
                    self.move_to(point()?)?;
                }
                self.press_keys(input.key.as_deref())?;
                for _ in 0..pixels.div_ceil(100) {
                    self.press_button(button)?;
                    self.release_button(button)?;
                    self.wait(10)?;
                }
                self.release()?;
            }
            _ => return Err(format!("unsupported desktop method: {method}").into()),
        }
        Ok(Value::Null)
    }
}
impl Drop for Desktop {
    fn drop(&mut self) {
        if let Err(error) = self.release() {
            eprintln!("input release failed: {error}");
        }
    }
}
fn main() -> Result<()> {
    let cancelled = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGTERM, signal_hook::consts::SIGINT] {
        signal_hook::flag::register(signal, cancelled.clone())?;
    }
    let mut desktop = Desktop::connect(cancelled.clone())?;
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.len() > 256 * 1024 {
            return Err("desktop request is too large".into());
        }
        let request: Request = serde_json::from_str(&line)?;
        let result = serde_json::from_value::<Input>(request.input)
            .map_err(Into::into)
            .and_then(|input| desktop.execute(&request.method, input));
        let response = match result {
            Ok(value) => json!({"id":request.id, "result":value}),
            Err(error) => {
                desktop.release()?;
                json!({"id":request.id, "error":error.to_string()})
            }
        };
        println!("{response}");
        io::stdout().flush()?;
        if cancelled.load(Ordering::Relaxed) {
            break;
        }
    }
    desktop.release()
}
