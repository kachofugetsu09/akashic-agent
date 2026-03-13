#!/bin/bash

# 安装 napcat
if [ ! -f "napcat/napcat.mjs" ]; then
    unzip -q NapCat.Shell.zip -d ./NapCat.Shell
    cp -rf NapCat.Shell/* napcat/
    rm -rf ./NapCat.Shell
fi
if [ ! -f "napcat/config/napcat.json" ]; then
    unzip -q NapCat.Shell.zip -d ./NapCat.Shell
    cp -rf NapCat.Shell/config/* napcat/config/
    rm -rf ./NapCat.Shell
fi

# 配置 WebUI Token
CONFIG_PATH=/app/napcat/config/webui.json

if [ ! -f "${CONFIG_PATH}" ] && [ -n "${WEBUI_TOKEN}" ]; then
    echo "正在配置 WebUI Token..."
    cat > "${CONFIG_PATH}" << EOF2
{
    "host": "0.0.0.0",
    "prefix": "${WEBUI_PREFIX}",
    "port": 6099,
    "token": "${WEBUI_TOKEN}",
    "loginRate": 3
}
EOF2
fi

if [ -n "${MODE}" ]; then
    cp /app/templates/$MODE.json /app/napcat/config/onebot11.json
fi

# 改为 :99，避免与桌面常用 :1 冲突
rm -rf "/tmp/.X99-lock"

: ${NAPCAT_GID:=0}
: ${NAPCAT_UID:=0}
usermod -o -u ${NAPCAT_UID} napcat
groupmod -o -g ${NAPCAT_GID} napcat
usermod -g ${NAPCAT_GID} napcat
chown -R ${NAPCAT_UID}:${NAPCAT_GID} /app

gosu napcat Xvfb :99 -screen 0 1080x760x16 +extension GLX +render -nolisten tcp > /dev/null 2>&1 &
sleep 2

export FFMPEG_PATH=/usr/bin/ffmpeg
export DISPLAY=:99
cd /app/napcat
if [ -n "${ACCOUNT}" ]; then
    exec gosu napcat /opt/QQ/qq --no-sandbox -q "$ACCOUNT"
else
    exec gosu napcat /opt/QQ/qq --no-sandbox
fi
