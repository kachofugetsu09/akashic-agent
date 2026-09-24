import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from bootstrap import web_auth
from bootstrap.web_auth import SESSION_COOKIE, WebAuth, hash_password, safe_next
from bootstrap.web_shell import create_web_shell_app


@pytest.fixture(scope="module")
def password_hash() -> str:
    return hash_password("correct horse")


def _client(tmp_path, auth: WebAuth | None) -> TestClient:
    app = create_web_shell_app(tmp_path / "config.toml", tmp_path, auth=auth)
    return TestClient(app, base_url="http://akashic.test")


def test_disabled_auth_keeps_existing_entry_open(tmp_path):
    with _client(tmp_path, None) as client:
        assert client.get("/api/auth/status").json() == {"auth_required": False, "authenticated": True}
        assert client.get("/api/shell/state").status_code == 200
        assert client.get("/login", follow_redirects=False).headers["location"] == "/"


def test_unauthenticated_page_api_and_websocket_are_rejected(tmp_path, password_hash):
    with _client(tmp_path, WebAuth(password_hash)) as client:
        page = client.get("/dashboard?x=1", headers={"accept": "text/html"}, follow_redirects=False)
        assert page.status_code == 303
        assert page.headers["location"] == "/login?next=%2Fdashboard%3Fx%3D1"
        assert client.get("/api/shell/state").status_code == 401
        assert client.get("/api/chat/sessions").status_code == 401
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect("/ws"):
                pass
        assert closed.value.code == web_auth.WEBSOCKET_UNAUTHORIZED
        assert client.get("/login").status_code == 200


def test_login_cookie_grants_access_and_forgery_or_password_change_revokes(tmp_path, password_hash, monkeypatch):
    monkeypatch.setattr(web_auth, "_FAILURE_DELAY_SECONDS", 0)
    with _client(tmp_path, WebAuth(password_hash)) as client:
        assert client.post("/api/auth/login", json={"password": "wrong"}).status_code == 401
        signed_in = client.post("/api/auth/login", json={"password": "correct horse"})
        assert signed_in.status_code == 200
        cookie = signed_in.headers["set-cookie"]
        assert "HttpOnly" in cookie and "SameSite=Lax" in cookie and "Secure" not in cookie
        assert client.get("/api/shell/state").status_code == 200
        token = client.cookies[SESSION_COOKIE]

        client.cookies.clear()
        client.cookies.set(SESSION_COOKIE, token[:-2] + ("AA" if not token.endswith("AA") else "BB"))
        assert client.get("/api/shell/state").status_code == 401

        client.cookies.clear()
        assert client.post("/api/auth/login", json={"password": "correct horse"}).status_code == 200
        assert client.get("/api/shell/state").status_code == 200
        logout = client.post("/api/auth/logout")
        assert "Max-Age=0" in logout.headers["set-cookie"]
        assert client.get("/api/shell/state").status_code == 401

    with _client(tmp_path, WebAuth(hash_password("new password"))) as rotated:
        rotated.cookies.set(SESSION_COOKIE, token)
        assert rotated.get("/api/shell/state").status_code == 401


def test_repeated_failures_are_throttled_and_cross_origin_login_rejected(tmp_path, password_hash, monkeypatch):
    monkeypatch.setattr(web_auth, "_FAILURE_DELAY_SECONDS", 0)
    with _client(tmp_path, WebAuth(password_hash)) as client:
        foreign = client.post(
            "/api/auth/login", json={"password": "correct horse"}, headers={"origin": "http://evil.test"},
        )
        assert foreign.status_code == 403
        for _ in range(web_auth._THROTTLE_MAX_FAILURES):
            assert client.post("/api/auth/login", json={"password": "nope"}).status_code == 401
        locked = client.post("/api/auth/login", json={"password": "correct horse"})
        assert locked.status_code == 429
        assert int(locked.headers["retry-after"]) > 0


def test_expired_session_is_rejected_and_old_session_is_refreshed(password_hash):
    auth = WebAuth(password_hash)
    token = auth.issue(1_000)
    assert auth.verify(token, 1_000 + web_auth.SESSION_TTL_SECONDS) is None
    assert auth.verify(token, 1_001) is not None
    with pytest.raises(ValueError):
        WebAuth("scrypt:bad")


def test_login_redirect_target_stays_on_this_site():
    assert safe_next("/dashboard?s=1") == "/dashboard?s=1"
    for hostile in ("https://evil.test", "//evil.test", "/\\evil.test", None, ""):
        assert safe_next(hostile) == "/"
