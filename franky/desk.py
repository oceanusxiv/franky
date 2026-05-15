from __future__ import annotations

import base64
import configparser
import hashlib
import http.client
import json
import logging
import os
import ssl
import time
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from http.client import HTTPSConnection, HTTPResponse
from typing import Optional, Any, Literal

import websockets.sync.client as ws_sync

logger = logging.getLogger(__name__)


def _user_config_home() -> str:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home and os.path.isabs(config_home):
        return config_home
    return os.path.expanduser("~/.config")


TOKEN_STORAGE_PATH = os.path.join(_user_config_home(), "franky", "control_tokens.conf")


class DeskError(Exception):
    pass


class FrankaAPIError(DeskError):
    def __init__(
        self,
        path: str,
        http_code: int,
        http_reason: str,
        headers: dict[str, str],
        message: str,
    ):
        super().__init__(
            f"Franka API returned error {http_code} ({http_reason}) when accessing end-point {path}: {message}"
        )
        self.path = path
        self.http_code = http_code
        self.headers = headers
        self.message = message


class TakeControlTimeoutError(DeskError):
    pass


class PilotButton(Enum):
    """Buttons on the Franka Desk Pilot interface."""

    CIRCLE = "circle"
    CROSS = "cross"
    CHECK = "check"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


class OperatingMode(Enum):
    EXECUTION = "Execution"
    PROGRAMMING = "Programming"


class BrakeState(Enum):
    LOCKED = "Locked"
    UNLOCKED = "Unlocked"


class NoTokenIdType:
    pass


NO_TOKEN_ID = NoTokenIdType()


@dataclass(frozen=True)
class PilotButtonEvent:
    """A single Pilot button state change."""

    button: PilotButton
    pressed: bool

    def __repr__(self) -> str:
        action = "pressed" if self.pressed else "released"
        return f"PilotButtonEvent({self.button.value} {action})"


@dataclass(frozen=True)
class _ControlToken:
    id: str | NoTokenIdType
    token: str


class _ControlTokenStore:
    _NO_TOKEN_ID = "__NO_TOKEN_ID__"

    def __init__(self, path: str):
        self._path = os.path.expanduser(path)

    def _section(self, api: str, hostname: str, username: str) -> str:
        return f"{api}:{hostname}:{username}"

    def _write(self, config: configparser.ConfigParser) -> None:
        directory = os.path.dirname(self._path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        fd = os.open(self._path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as config_file:
            config.write(config_file)
        os.chmod(self._path, 0o600)

    def load(self, api: str, hostname: str, username: str) -> _ControlToken | None:
        config = configparser.ConfigParser()
        if not os.path.exists(self._path):
            return None
        config.read(self._path)
        section = self._section(api, hostname, username)
        if not config.has_section(section):
            return None
        token_id = config.get(section, "id", fallback="")
        token = config.get(section, "token", fallback="")
        if not token_id or not token:
            return None
        if token_id == self._NO_TOKEN_ID:
            return _ControlToken(id=NO_TOKEN_ID, token=token)
        return _ControlToken(id=token_id, token=token)

    def save(
        self, api: str, hostname: str, username: str, control_token: _ControlToken
    ) -> None:
        config = configparser.ConfigParser()
        if os.path.exists(self._path):
            config.read(self._path)
        token_id = (
            self._NO_TOKEN_ID
            if control_token.id is NO_TOKEN_ID
            else str(control_token.id)
        )
        config[self._section(api, hostname, username)] = {
            "id": token_id,
            "token": control_token.token,
        }
        self._write(config)

    def delete(self, api: str, hostname: str, username: str) -> None:
        if not os.path.exists(self._path):
            return
        config = configparser.ConfigParser()
        config.read(self._path)
        if not config.remove_section(self._section(api, hostname, username)):
            return
        self._write(config)


def _make_control_token_store(
    token_storage: bool | str | os.PathLike,
) -> _ControlTokenStore | None:
    if token_storage is False:
        return None
    if token_storage is True:
        return _ControlTokenStore(TOKEN_STORAGE_PATH)
    return _ControlTokenStore(os.fspath(token_storage))


def _encode_password(user: str, password: str) -> str:
    bs = ",".join(
        [
            str(b)
            for b in hashlib.sha256(
                (password + "#" + user + "@franka").encode("utf-8")
            ).digest()
        ]
    )
    return base64.encodebytes(bs.encode("utf-8")).decode("utf-8")


class BaseDesk(ABC):
    def __init__(
        self,
        hostname: str,
        username: str,
        password: str,
        token_storage: bool | str | os.PathLike = False,
        token_store_api: str = "",
    ):
        super().__init__()
        self.__hostname = hostname
        self.__username = username
        self.__password = password
        self.__token_store_api = token_store_api
        self.__token_store = _make_control_token_store(token_storage)
        self.__pilot_button_socket = None
        self.__client = None
        stored_token = self.__load_control_token()
        self.__control_token = stored_token.token if stored_token is not None else None
        self.__control_token_id = stored_token.id if stored_token is not None else None

    def open(self, timeout: float = 30.0):
        if self.is_open:
            raise RuntimeError("Session is already open.")
        self.__client = HTTPSConnection(
            self.hostname, timeout=timeout, context=ssl._create_unverified_context()
        )
        self.__client.connect()

    def close(self):
        if not self.is_open:
            raise RuntimeError("Session is not open.")
        try:
            self._close_pilot_button_socket()
            if self.__control_token is not None:
                if self.has_control:
                    self.release_control()
                else:
                    self.__clear_control_token(delete_from_store=True)
        finally:
            self._close_client()

    def take_control(self, wait_timeout: float = 30.0, force: bool = False):
        if not self.has_control:
            self.__control_token_id, self.__control_token = self._take_control(
                wait_timeout=wait_timeout, force=force
            )
            self.__save_control_token()

    def release_control(self):
        if self.__control_token is not None:
            self._release_control()
        self.__clear_control_token(delete_from_store=True)

    def send_api_request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        content: dict[str, Any] | None = None,
        content_encoding: Literal["json", "x-www-form-urlencoded"] = "json",
        method: Literal["GET", "POST", "DELETE"] = "POST",
        response_encoding: Optional[Literal["json", "text"]] = None,
    ) -> Any:
        return self.__parse_response(
            *self.__request(
                path,
                method=method,
                **self.__encode_content(
                    headers=headers, content=content, content_encoding=content_encoding
                ),
            ),
            response_encoding=response_encoding,
        )

    def send_control_api_request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        content: dict[str, Any] | None = None,
        content_encoding: Literal["json", "x-www-form-urlencoded"] = "json",
        method: Literal["GET", "POST", "DELETE"] = "POST",
        response_encoding: Optional[Literal["json", "text"]] = None,
    ) -> Any:
        return self.__parse_response(
            *self.__request_with_control(
                path,
                method=method,
                **self.__encode_content(
                    headers=headers, content=content, content_encoding=content_encoding
                ),
            ),
            response_encoding=response_encoding,
        )

    def send_raw_api_request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ) -> bytes:
        return self.__request(path, headers, body, method)[0]

    def send_raw_control_api_request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ) -> bytes:
        return self.__request_with_control(path, headers, body, method)[0]

    def __request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ) -> tuple[bytes, Optional[str]]:
        last_error = None
        for i in range(3):
            try:
                return self._send_raw_api_request(path, headers, body, method)
            except http.client.RemoteDisconnected as ex:
                last_error = ex
        raise last_error

    def __request_with_control(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ) -> tuple[bytes, Optional[str]]:
        if headers is None:
            headers = {}
        self.__check_control_token()
        _headers = {"X-Control-Token": self.__control_token}
        _headers.update(headers)
        return self.__request(path, headers=_headers, method=method, body=body)

    def poll_buttons(self, timeout: Optional[float] = 0.0) -> list[PilotButtonEvent]:
        """Poll for Pilot button events, returning all currently available.

        Args:
            timeout: Maximum seconds to wait for the first event.
                0.0 (default) returns immediately with any buffered events.
                None blocks until at least one event is available.
                After the first event arrives, any additional buffered
                events are drained without waiting.

        Returns:
            List of button events, or an empty list if none arrived
            within the timeout.
        """
        if not self.is_open:
            raise RuntimeError(
                "Session is not open. Call open() or use a context manager first."
            )

        try:
            websocket = self._get_pilot_button_socket()
            message = websocket.recv(timeout=timeout)
        except TimeoutError:
            return []
        except Exception:
            self._close_pilot_button_socket()
            raise

        events = list(self.__parse_pilot_button_payload(message))
        while True:
            try:
                message = websocket.recv(timeout=0.0)
            except TimeoutError:
                return events
            except Exception:
                self._close_pilot_button_socket()
                raise
            events.extend(self.__parse_pilot_button_payload(message))

    @abstractmethod
    def _get_pilot_auth_headers(self) -> dict[str, str]:
        pass

    @abstractmethod
    def _get_default_headers(self) -> dict[str, str]:
        pass

    @abstractmethod
    def _take_control(
        self, wait_timeout: float = 30.0, force: bool = False
    ) -> tuple[str, str]:
        pass

    @abstractmethod
    def _get_has_control(self):
        pass

    @abstractmethod
    def _release_control(self):
        pass

    @abstractmethod
    def _get_system_status(self) -> dict[str, Any]:
        pass

    @abstractmethod
    def _get_operating_mode(self) -> OperatingMode:
        pass

    @abstractmethod
    def _get_brake_state(self) -> BrakeState:
        pass

    @abstractmethod
    def enable_fci(self):
        pass

    @abstractmethod
    def unlock_brakes(self):
        pass

    @abstractmethod
    def lock_brakes(self):
        pass

    @abstractmethod
    def set_mode_programming(self):
        pass

    @abstractmethod
    def set_mode_execution(self):
        pass

    @abstractmethod
    def execute_self_test(self):
        pass

    def _send_raw_api_request(
        self,
        path: str,
        headers: Optional[dict[str, str]] = None,
        body: Optional[Any] = None,
        method: Literal["GET", "POST", "DELETE"] = "POST",
    ) -> tuple[bytes, Optional[str]]:
        _headers = self._get_default_headers()
        if headers is not None:
            _headers.update(headers)
        self.__client.request(method, path, headers=_headers, body=body)
        res: HTTPResponse = self.__client.getresponse()
        if res.status not in (200, 204):
            raise FrankaAPIError(
                path,
                res.status,
                res.reason,
                dict(res.headers),
                res.read().decode("utf-8"),
            )
        return res.read(), res.getheader("Content-Type")

    @staticmethod
    def __parse_response(
        body: bytes,
        content_type: Optional[str],
        response_encoding: Optional[Literal["json", "text"]] = None,
    ) -> Any:
        if not body:
            return None
        if response_encoding is None:
            media_type = (content_type or "").split(";", 1)[0].strip().lower()
            if media_type == "application/json" or media_type.endswith("+json"):
                response_encoding = "json"
            else:
                response_encoding = "text"
        if response_encoding == "json":
            return json.loads(body)
        return body.decode("utf-8")

    @staticmethod
    def __encode_content(
        headers: Optional[dict[str, str]] = None,
        content: dict[str, Any] | None = None,
        content_encoding: Literal["json", "x-www-form-urlencoded"] = "json",
    ):
        if headers is None:
            headers = {"content-type": f"application/{content_encoding}"}
        if content is None:
            body = None
        else:
            if content_encoding == "json":
                body = json.dumps(content)
            elif content_encoding == "x-www-form-urlencoded":
                body = urllib.parse.urlencode(content)
            else:
                raise ValueError(f"Unsupported content encoding: {content_encoding}")
        return {"body": body, "headers": headers}

    def __check_control_token(self):
        if self.__control_token is None:
            raise RuntimeError(
                "Client does not have control. Call take_control() first."
            )

    def __load_control_token(self) -> _ControlToken | None:
        if self.__token_store is None:
            return None
        return self.__token_store.load(
            self.__token_store_api, self.__hostname, self.__username
        )

    def __save_control_token(self) -> None:
        if (
            self.__token_store is None
            or self.__control_token_id is None
            or self.__control_token is None
        ):
            return
        self.__token_store.save(
            self.__token_store_api,
            self.__hostname,
            self.__username,
            _ControlToken(id=self.__control_token_id, token=self.__control_token),
        )

    def __clear_control_token(self, delete_from_store: bool = False) -> None:
        self.__control_token = None
        self.__control_token_id = None
        if delete_from_store and self.__token_store is not None:
            self.__token_store.delete(
                self.__token_store_api, self.__hostname, self.__username
            )

    def _close_client(self) -> None:
        if self.__client is not None:
            self.__client.close()
            self.__client = None

    def _get_pilot_button_socket(self):
        if self.__pilot_button_socket is None:
            uri = f"wss://{self.__hostname}/desk/api/navigation/events"
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ssl_ctx.check_hostname = False
            ssl_ctx.verify_mode = ssl.CERT_NONE
            self.__pilot_button_socket = ws_sync.connect(
                uri,
                ssl=ssl_ctx,
                additional_headers=self._get_pilot_auth_headers(),
                open_timeout=2.0,
            )
        return self.__pilot_button_socket

    def _close_pilot_button_socket(self) -> None:
        if self.__pilot_button_socket is None:
            return
        try:
            self.__pilot_button_socket.close()
        finally:
            self.__pilot_button_socket = None

    @staticmethod
    def __parse_pilot_button_payload(payload: str) -> list[PilotButtonEvent]:
        data = json.loads(payload)
        if not isinstance(data, dict):
            logger.debug(
                "Ignoring non-object event payload of type %s: %r",
                type(data).__name__,
                data,
            )
            return []
        events = []
        for key, value in data.items():
            button = PilotButton(key)
            events.append(PilotButtonEvent(button=button, pressed=bool(value)))
        return events

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @property
    def is_open(self) -> bool:
        return self.__client is not None

    @property
    def hostname(self) -> str:
        return self.__hostname

    @property
    def username(self) -> str:
        return self.__username

    @property
    def password(self) -> str:
        return self.__password

    @property
    def control_token(self) -> str | None:
        return self.__control_token

    @property
    def control_token_id(self) -> str | None:
        return self.__control_token_id

    @property
    def client(self) -> HTTPSConnection:
        return self.__client

    @property
    def has_control(self):
        if self.__control_token_id is not None:
            return self._get_has_control()
        return False

    @property
    def system_status(self) -> dict[str, Any]:
        return self._get_system_status()

    @property
    def operating_mode(self) -> OperatingMode:
        return self._get_operating_mode()

    @property
    def brake_state(self) -> BrakeState:
        return self._get_brake_state()


class DeskWebSession(BaseDesk):
    """Desk web session for the legacy Franka Desk API.

    Compatible with Panda and FR3 on pre-v1 firmware. For FR3 on current
    firmware, use :class:`Desk` instead.
    """

    def __init__(
        self,
        hostname: str,
        username: str,
        password: str,
        token_storage: bool | str | os.PathLike = False,
    ):
        super().__init__(
            hostname,
            username,
            password,
            token_storage=token_storage,
            token_store_api="legacy",
        )
        self.__token = None

    def _get_pilot_auth_headers(self) -> dict[str, str]:
        return {"authorization": self.__token}

    def _get_default_headers(self) -> dict[str, str]:
        headers = {}
        if self.__token is not None:
            headers["Cookie"] = f"authorization={self.__token}"
        return headers

    def open(self, timeout: float = 30.0):
        super().open(timeout=timeout)
        try:
            self.__token = self.send_api_request(
                "/admin/api/login",
                content={
                    "login": self.username,
                    "password": _encode_password(self.username, self.password),
                },
                response_encoding="text",
            )
        except:
            self._close_client()
            raise

    def close(self):
        super().close()
        self.__token = None

    def _take_control(self, wait_timeout: float = 30.0, force: bool = False):
        response_dict = self.send_api_request(
            f"/admin/api/control-token/request{'?force' if force else ''}",
            content={"requestedBy": self.username},
        )
        if force:
            print(
                "Forcibly taking control: "
                f"Please physically take control by pressing the top button on the FR3 within {wait_timeout}s!"
            )
        token_id, token = response_dict["id"], response_dict["token"]

        # Cannot use self.has_control here as the token is only stored in the
        # base class once this method returns.
        def granted() -> bool:
            active_token = self._get_system_status()["controlToken"]["activeToken"]
            return active_token is not None and str(active_token["id"]) == str(token_id)

        start = time.time()
        has_control = granted()
        while time.time() - start < wait_timeout and not has_control:
            time.sleep(max(0.0, min(1.0, wait_timeout - (time.time() - start))))
            has_control = granted()
        if not has_control:
            raise TakeControlTimeoutError(
                f"Timed out waiting for control to be granted after {wait_timeout}s."
            )
        return str(token_id), token

    def _get_has_control(self):
        status = self._get_system_status()
        active_token = status["controlToken"]["activeToken"]
        return (
            active_token is not None
            and str(active_token["id"]) == self.control_token_id
        )

    def _release_control(self):
        self.send_control_api_request(
            "/admin/api/control-token",
            method="DELETE",
            content={"token": self.control_token},
        )

    def _get_operating_mode(self) -> OperatingMode:
        return OperatingMode(self._get_system_status()["derived"]["operatingMode"])

    def _get_brake_state(self) -> BrakeState:
        status = self._get_system_status()
        if all(b == "Unlocked" for b in status["safety"]["brakeState"]):
            return BrakeState.UNLOCKED
        elif all(b == "Locked" for b in status["safety"]["brakeState"]):
            return BrakeState.LOCKED
        else:
            raise DeskError(
                "Inconsistent brake state: some joints are locked while others are unlocked."
            )

    def enable_fci(self):
        self.send_control_api_request(
            "/desk/api/system/fci",
            content={
                "token": base64.b64encode(self.control_token.encode("ascii")).decode(
                    "ascii"
                )
            },
            content_encoding="x-www-form-urlencoded",
        )

    def start_task(self, task: str):
        self.send_api_request(
            "/desk/api/execution",
            content={"id": task},
            content_encoding="x-www-form-urlencoded",
        )

    def unlock_brakes(self):
        self.send_control_api_request(
            "/desk/api/joints/unlock",
            content_encoding="x-www-form-urlencoded",
        )

    def lock_brakes(self):
        self.send_control_api_request(
            "/desk/api/joints/lock",
            content_encoding="x-www-form-urlencoded",
        )

    def set_mode_programming(self):
        self.send_control_api_request(
            "/desk/api/operating-mode/programming",
            content_encoding="x-www-form-urlencoded",
        )

    def set_mode_execution(self):
        self.send_control_api_request(
            "/desk/api/operating-mode/execution",
            content_encoding="x-www-form-urlencoded",
        )

    def _get_system_status(self):
        return self.send_api_request("/admin/api/system-status", method="GET")

    def execute_self_test(self):
        if self._get_system_status()["safety"]["recoverableErrors"]["td2Timeout"]:
            self.send_control_api_request(
                "/admin/api/safety/recoverable-safety-errors/acknowledge?error_id=TD2Timeout"
            )
        response = self.send_control_api_request(
            "/admin/api/safety/td2-tests/execute",
        )
        assert response["code"] == "SuccessResponse"
        time.sleep(0.5)
        while (
            self._get_system_status()["safety"]["safetyControllerStatus"] == "SelfTest"
        ):
            time.sleep(0.5)

    @property
    def token(self) -> str:
        return self.__token


class Desk(BaseDesk):
    """Robot web session for the current Franka Desk API (v1).

    Uses HTTP Basic authentication on every request. Compatible with FR3
    on System 5+ firmware. For older firmware, use :class:`DeskWebSession`.
    """

    def __init__(
        self,
        hostname: str,
        username: str,
        password: str,
        token_storage: bool | str | os.PathLike = False,
    ):
        super().__init__(
            hostname,
            username,
            password,
            token_storage=token_storage,
            token_store_api="v1",
        )
        self.__warned_no_token_id = False

    def _get_pilot_auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Basic {self.__auth_string}"}

    def _get_default_headers(self) -> dict[str, str]:
        return {"Authorization": f"Basic {self.__auth_string}"}

    def open(self, timeout: float = 30.0):
        super().open(timeout=timeout)
        try:
            # Verify connectivity and credentials right away.
            self._get_system_status()
        except:
            self._close_client()
            raise

    def _take_control(self, wait_timeout: float = 30.0, force: bool = False):
        # force is ignored on the v1 API -- the server queues the request
        # (up to wait_timeout seconds) until the current holder releases.
        res = self.send_api_request(
            "/api/system/control-token:take",
            content={"owner": self.username, "timeout": int(wait_timeout)},
        )
        return res.get("tokenId", NO_TOKEN_ID), res["token"]

    def _release_control(self):
        self.send_control_api_request("/api/system/control-token:release")

    def _get_has_control(self) -> bool:
        data = self.send_api_request("/api/system/control-token", method="GET")
        if self.control_token_id == NO_TOKEN_ID:
            if not self.__warned_no_token_id:
                logger.warning(
                    "Your Franka API does not support checking if the current token is still valid. Falling back to "
                    "checking for just the owner. Upgrade the Franka firmware or resort to franky.DeskWebSession to "
                    "resolve this issue."
                )
                self.__warned_no_token_id = True
            return data.get("owner") == self.username
        return str(data.get("tokenId")) == str(self.control_token_id)

    def _get_operating_mode(self) -> OperatingMode:
        data = self.send_api_request("/api/system/operating-mode", method="GET")
        return OperatingMode(data["status"])

    def _get_brake_state(self) -> BrakeState:
        data = self.send_api_request("/api/arm/joints", method="GET")
        if all(j["brakeStatus"] == "Unlocked" for j in data):
            return BrakeState.UNLOCKED
        elif all(j["brakeStatus"] == "Locked" for j in data):
            return BrakeState.LOCKED
        else:
            raise DeskError(
                "Inconsistent brake state: some joints are locked while others are unlocked."
            )

    def enable_fci(self):
        self.send_control_api_request("/api/fci:activate")

    def disable_fci(self):
        self.send_control_api_request("/api/fci:deactivate")

    def get_fci_status(self) -> str:
        return self.send_api_request("/api/fci", method="GET")["status"]

    def unlock_brakes(self):
        self.send_control_api_request("/api/arm/joints:unlock")

    def lock_brakes(self):
        self.send_control_api_request("/api/arm/joints:lock")

    def set_mode_programming(self):
        self.send_control_api_request(
            "/api/system/operating-mode:change",
            content={"desiredOperatingMode": "Programming"},
        )

    def set_mode_execution(self):
        self.send_control_api_request(
            "/api/system/operating-mode:change",
            content={"desiredOperatingMode": "Execution"},
        )

    def _get_system_status(self) -> dict:
        return self.send_api_request("/api/system", method="GET")

    def execute_self_test(self):
        self.send_control_api_request("/api/safety/self-tests:execute")
        while (
            self.send_api_request("/api/safety/self-tests", method="GET")["status"]
            == "Running"
        ):
            time.sleep(0.5)

    def get_recovery_status(self) -> Optional[dict]:
        """Return the active recovery descriptor, or None if no recovery is needed."""
        data = self.send_api_request("/api/safety/recovery", method="GET")
        return data.get("recovery")

    def recover(self) -> None:
        """Attempt to clear an active recoverable safety error.

        Handles errors that can be confirmed programmatically
        (LifetimeStatusExceeded, SelfTestsElapsed, GenericJointError,
        SafetyError, SafeInputUnacknowledged, SafetyRuleViolation).
        Raises :class:`DeskError` for errors that require
        physical joint movement (JointPositionError, JointLimitViolation).
        Does nothing if no recovery is needed.
        """
        recovery = self.get_recovery_status()
        if recovery is None:
            return
        recovery_type = recovery["type"]
        if recovery_type in ("JointPositionError", "JointLimitViolation"):
            raise DeskError(
                f"Recovery type '{recovery_type}' requires physical joint movement. "
                "Move the affected joints to their reference positions and confirm "
                "via the Desk interface. Consult the product manual for details."
            )
        content: dict[str, Any] = {"type": recovery_type}
        if recovery_type == "SafetyError":
            content["safetyErrors"] = recovery.get("safetyErrors", [])
        elif recovery_type == "SafeInputUnacknowledged":
            content["safeInputs"] = recovery.get("safeInputs", [])
        self.send_api_request("/api/safety/recovery:confirm", content=content)

    def reboot(self) -> None:
        """Initiate a system reboot. Closes all open connections."""
        self.send_api_request("/api/system:reboot")

    def shutdown(self) -> None:
        """Initiate a system shutdown."""
        self.send_api_request("/api/system:shutdown")

    @property
    def __auth_string(self):
        return base64.b64encode(f"{self.username}:{self.password}".encode()).decode()
