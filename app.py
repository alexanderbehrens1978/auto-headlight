#!/usr/bin/env python3
import os, time, threading, json, yaml, glob
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, abort

# Headless: unterbinde Qt/xcb-Fehler
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# ---- GPIO (failsafe, wenn nicht vorhanden) ----
GPIO_OK = True
try:
    from gpiozero import OutputDevice, Button
except Exception:
    GPIO_OK = False
    class OutputDevice:
        def __init__(self, *a, **k): self._v=False
        def on(self):  self._v=True
        def off(self): self._v=False
    class Button:
        def __init__(self, *a, **k): self._p=True
        @property
        def is_pressed(self): return not self._p

CFG_PATH = "/opt/auto-headlight/config.yaml"

APP = Flask(__name__)
detector = None
detector_lock = threading.Lock()

# ------------------ Config ------------------
def load_cfg():
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)

def save_cfg(cfg):
    with open(CFG_PATH, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

def deep_merge(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst

# ------------------ Devices -----------------
def list_video_nodes():
    vids = []
    for p in sorted(glob.glob("/dev/video*")):
        try:
            idx = int(p.replace("/dev/video",""))
            name_path = f"/sys/class/video4linux/video{idx}/name"
            if os.path.exists(name_path):
                with open(name_path, "r") as f:
                    name = f.read().strip()
            else:
                name = "Video Device"
            vids.append({"index": idx, "name": name})
        except Exception:
            continue
    return vids

def csi_available():
    # einfache Heuristik: libcamera auf dem Pi + Unicam vorhanden → CSI 0 nutzbar
    try:
        from picamera2 import Picamera2
        _ = Picamera2.global_camera_info()
        return True
    except Exception:
        return False

# ------------------ Utils -------------------
def roi_rect(img_shape, rel_rect):
    h, w = img_shape[:2]
    x1 = int(rel_rect[0] * w); y1 = int(rel_rect[1] * h)
    x2 = int(rel_rect[2] * w); y2 = int(rel_rect[3] * h)
    return max(0,x1), max(0,y1), min(w-1,x2), min(h-1,y2)

def mask_range(hsv, lower, upper):
    return cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))

def find_bright_spots(mask, min_area):
    if min_area <= 0: return []
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts=[]
    for c in contours:
        a = cv2.contourArea(c)
        if a >= min_area:
            (x,y),_ = cv2.minEnclosingCircle(c)
            pts.append((int(x),int(y),int(a)))
    return pts

class DebouncedFlag:
    def __init__(self, confirm_frames=3, hold_ms=800):
        self.cf = max(1,int(confirm_frames))
        self.hold = max(0,int(hold_ms))
        self.cnt = 0
        self.active=False
        self.t0=0.0
    def update(self, cond, now_ms):
        self.cnt = min(self.cnt+1,self.cf) if cond else max(self.cnt-1,0)
        want = (self.cnt>=self.cf)
        if want!=self.active and (now_ms - self.t0)>=self.hold:
            self.active=want
            self.t0=now_ms
        return self.active

# -------------- Capture-Factory --------------
def open_capture(cfg):
    """
    source_type:
      - "video": Datei
      - "camera": Index -> 0 = CSI(Picamera2), >0 = USB(OpenCV V4L2 + MJPG)
    Rückgabe: (cap_obj, read_fn, release_fn, desc)
    """
    st   = cfg["camera"].get("source_type","camera")
    idx  = int(cfg["camera"]["camera_index"])
    w    = int(cfg["camera"]["width"])
    h    = int(cfg["camera"]["height"])
    fps  = int(cfg["camera"]["fps"])
    vpth = cfg["camera"].get("video_path","")

    if st == "video" and vpth:
        cap = cv2.VideoCapture(vpth)
        if not cap.isOpened():
            return None, None, None, f"video open fail: {vpth}"
        def _read(): return cap.read()
        def _release(): cap.release()
        return cap, _read, _release, f"VIDEO:{os.path.basename(vpth)}"

    if st == "camera" and idx == 0:
        try:
            from picamera2 import Picamera2
        except Exception as e:
            print("❌ Picamera2 nicht verfügbar:", e)
            return None, None, None, "CSI(Picamera2 fehlt)"
        picam = Picamera2()
        video_cfg = picam.create_video_configuration(main={"size": (w, h), "format": "RGB888"})
        picam.configure(video_cfg)
        picam.start()
        time.sleep(0.1)
        def _read():
            fr = picam.capture_array()  # RGB
            return True, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR)
        def _release():
            try: picam.stop()
            except: pass
        return picam, _read, _release, f"CSI@{w}x{h}@{fps}"

    # USB
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS,          fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   2)
    if not cap.isOpened():
        return None, None, None, f"USB{idx} open fail"
    def _read(): return cap.read()
    def _release(): cap.release()
    return cap, _read, _release, f"USB@{idx} {w}x{h}@{fps}"

# -------------- Detector-Thread --------------
class DetectorThread(threading.Thread):
    def __init__(self, cfg_getter):
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self._running=False
        self.cfg_getter = cfg_getter
        self.camera_desc="n/a"
        # Stream-Puffer
        self._jpeg_lock = threading.Lock()
        self._last_jpeg = None
        # IO
        self.relay = None
        self.safety = None
        self.is_high = False

    def stop(self): self._stop.set()
    def is_running(self): return self._running
    def get_last_jpeg(self):
        with self._jpeg_lock: return self._last_jpeg

    def _setup_io(self, cfg):
        pin  = int(cfg["gpio"].get("highbeam_pin", -1))
        ah   = bool(cfg["gpio"].get("active_high", True))
        inv  = bool(cfg["gpio"].get("invert_relay", False))
        mode_high = ah ^ inv
        if pin >= 0:
            self.relay = OutputDevice(pin, active_high=mode_high, initial_value=False) if GPIO_OK else OutputDevice()
        sp = int(cfg["gpio"].get("safety_pin", -1))
        if sp >= 0 and GPIO_OK:
            try:
                self.safety = Button(sp, pull_up=True)
            except Exception:
                self.safety = None

    def _set_light(self, on:bool):
        self.is_high = on
        if self.relay is None: return
        self.relay.on() if on else self.relay.off()

    def run(self):
        cfg = self.cfg_getter()
        self._setup_io(cfg)

        cap, read_fn, release_fn, desc = open_capture(cfg)
        self.camera_desc = desc
        if cap is None:
            print("❌ Quelle konnte nicht geöffnet werden:", desc)
            return

        hi_hold = DebouncedFlag(cfg["timing"]["confirm_frames"], cfg["timing"]["hold_high_ms"])
        lo_hold = DebouncedFlag(cfg["timing"]["confirm_frames"], cfg["timing"]["hold_low_ms"])
        self._set_light(False)

        self._running=True
        try:
            while not self._stop.is_set():
                ok, frame = read_fn()
                if not ok or frame is None:
                    time.sleep(0.02); continue

                C = self.cfg_getter()
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # Thresholds
                t = C["thresholds"]
                white  = mask_range(hsv, t["white_min"],  t["white_max"])
                yellow = mask_range(hsv, t["yellow_min"], t["yellow_max"])
                r1 = mask_range(hsv, t["red1_min"], t["red1_max"])
                r2 = mask_range(hsv, t["red2_min"], t["red2_max"])
                red = cv2.bitwise_or(r1, r2)

                # ROI (oncoming)
                x1o,y1o,x2o,y2o = roi_rect(frame.shape, C["roi"]["oncoming"])
                roi_on = np.zeros_like(white); roi_on[y1o:y2o, x1o:x2o]=255
                white_o  = cv2.bitwise_and(white,  roi_on)
                yellow_o = cv2.bitwise_and(yellow, roi_on)
                red_o    = cv2.bitwise_and(red,    roi_on)

                # Konturen
                ct=C["contours"]
                oncoming = find_bright_spots(white_o,  ct["min_area_oncoming"])
                street   = find_bright_spots(yellow_o, ct["min_area_street"])
                tails    = find_bright_spots(red_o,    ct["min_area_tail"])

                # Entscheidung
                now_ms = time.time()*1000.0
                danger = (len(oncoming)>0) or (len(street)>0) or (len(tails)>0)

                low_state  = lo_hold.update(danger, now_ms)
                high_state = hi_hold.update(not danger, now_ms)

                safety_active = False
                if self.safety is not None:
                    safety_active = (not self.safety.is_pressed)  # Pull-Up -> gedrückt==False

                want_high = high_state and not low_state and not safety_active

                if want_high != self.is_high:
                    self._set_light(want_high)

                # Overlay
                disp = frame
                if C["ui"]["draw"]:
                    disp = frame.copy()
                    cv2.rectangle(disp,(x1o,y1o),(x2o,y2o),(0,255,255),1)
                    for (x,y,a) in oncoming: cv2.circle(disp,(x,y),5,(0,255,0),-1)
                    for (x,y,a) in street:   cv2.circle(disp,(x,y),5,(0,255,255),-1)
                    for (x,y,a) in tails:    cv2.circle(disp,(x,y),5,(255,0,0),-1)
                    txt = f"{'HIGH' if self.is_high else 'LOW'} danger:{danger} safety:{safety_active}"
                    cv2.putText(disp, txt, (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

                # JPEG in Stream-Puffer
                ok_jpg, jpg = cv2.imencode(".jpg", disp, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_jpg:
                    with self._jpeg_lock:
                        self._last_jpeg = jpg.tobytes()

        finally:
            try: release_fn()
            except: pass
            try: cv2.destroyAllWindows()
            except: pass
            self._set_light(False)
            self._running=False

# --------------- Flask Routes ---------------
@APP.route("/")
def index():
    cfg = load_cfg()
    with detector_lock:
        running = detector.is_running() if detector else False
        camdesc = detector.camera_desc if detector else "n/a"

    devices = {
        "csi_available": csi_available(),
        "video_nodes": list_video_nodes(),   # [{'index':1,'name':'...'}, ...]
    }

    return render_template("index.html",
                           cfg=cfg, running=running,
                           state={"camera": camdesc},
                           devices=devices)

@APP.route("/apply", methods=["POST"])
def apply():
    cfg = load_cfg()
    f = request.form

    def num(name, typ=float, default=None):
        v = f.get(name, None)
        if v is None or v == "": return default
        try: return typ(v)
        except: return default

    def booly(name, default=False):
        v = f.get(name, None)
        if v is None: return default
        return str(v).lower() in ("1","true","on","yes")

    # camera
    cfg["camera"]["source_type"]  = f.get("camera.source_type", cfg["camera"]["source_type"])
    cfg["camera"]["camera_index"] = int(num("camera.camera_index", int, cfg["camera"]["camera_index"]))
    cfg["camera"]["video_path"]   = f.get("camera.video_path", cfg["camera"]["video_path"])
    cfg["camera"]["width"]        = int(num("camera.width", int, cfg["camera"]["width"]))
    cfg["camera"]["height"]       = int(num("camera.height", int, cfg["camera"]["height"]))
    cfg["camera"]["fps"]          = int(num("camera.fps", int, cfg["camera"]["fps"]))

    # roi
    for name in ("oncoming","near"):
        for i in range(4):
            key = f"roi.{name}.{i}"
            val = num(key, float, cfg["roi"][name][i])
            cfg["roi"][name][i] = float(val)

    # thresholds
    def set_triplet(base):
        for i in range(3):
            key = f"thresholds.{base}.{i}"
            cfg["thresholds"][base][i] = int(num(key, int, cfg["thresholds"][base][i]))
    for k in ("white_min","white_max","yellow_min","yellow_max","red1_min","red1_max","red2_min","red2_max"):
        set_triplet(k)

    # contours
    for k in ("min_area_oncoming","min_area_street","min_area_tail"):
        cfg["contours"][k] = int(num(f"contours.{k}", int, cfg["contours"][k]))

    # timing
    cfg["timing"]["confirm_frames"] = int(num("timing.confirm_frames", int, cfg["timing"]["confirm_frames"]))
    cfg["timing"]["hold_high_ms"]   = int(num("timing.hold_high_ms", int, cfg["timing"]["hold_high_ms"]))
    cfg["timing"]["hold_low_ms"]    = int(num("timing.hold_low_ms", int, cfg["timing"]["hold_low_ms"]))

    # gpio
    cfg["gpio"]["highbeam_pin"] = int(num("gpio.highbeam_pin", int, cfg["gpio"]["highbeam_pin"]))
    cfg["gpio"]["safety_pin"]   = int(num("gpio.safety_pin", int, cfg["gpio"]["safety_pin"]))
    cfg["gpio"]["active_high"]  = booly("gpio.active_high", cfg["gpio"]["active_high"])
    cfg["gpio"]["invert_relay"] = booly("gpio.invert_relay", cfg["gpio"]["invert_relay"])

    # ui
    cfg["ui"]["draw"]        = booly("ui.draw", cfg["ui"]["draw"])
    cfg["ui"]["show_window"] = booly("ui.show_window", cfg["ui"]["show_window"])

    save_cfg(cfg)
    return redirect(url_for("index"))

@APP.route("/switch", methods=["POST"])
def switch_camera():
    """
    Schaltet bequem zwischen CSI (index 0) und USB (/dev/videoX) um.
    - setzt camera.source_type='camera'
    - setzt camera.camera_index auf gewünschten Index
    - wenn Detector läuft: sauber neu starten
    """
    target = request.form.get("camera_index", "").strip()
    if target == "":
        return redirect(url_for("index"))
    try:
        idx = int(target)
    except ValueError:
        return "invalid camera_index", 400

    cfg = load_cfg()
    cfg["camera"]["source_type"] = "camera"
    cfg["camera"]["camera_index"] = idx
    save_cfg(cfg)

    # laufenden Detector neu starten
    global detector
    with detector_lock:
        was_running = detector.is_running() if detector else False
        if was_running:
            detector.stop()
            detector.join(timeout=3)
            detector = DetectorThread(load_cfg)
            detector.start()
    return redirect(url_for("index"))

@APP.route("/start", methods=["POST"])
def start_detection():
    global detector
    with detector_lock:
        if detector and detector.is_running():
            return redirect(url_for("index"))
        detector = DetectorThread(load_cfg)
        detector.start()
    return redirect(url_for("index"))

@APP.route("/stop", methods=["POST"])
def stop_detection():
    global detector
    with detector_lock:
        if detector and detector.is_running():
            detector.stop()
            detector.join(timeout=3)
        detector = None
    return redirect(url_for("index"))

@APP.route("/status")
def status():
    with detector_lock:
        running = detector.is_running() if detector else False
        cam = detector.camera_desc if detector else "n/a"
    return jsonify({"running": running, "camera": cam, "devices":{
        "csi_available": csi_available(), "video_nodes": list_video_nodes()
    }})

@APP.route("/stream")
def stream():
    with detector_lock:
        d = detector
    if d is None or not d.is_running():
        return Response("Detector not running", status=503)

    def gen():
        t0=time.time()
        while d.get_last_jpeg() is None and (time.time()-t0)<5:
            time.sleep(0.05)
        while True:
            frame = d.get_last_jpeg()
            if not frame:
                time.sleep(0.05); continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n" +
                   frame + b"\r\n")
            fps = max(1, min(20, int(load_cfg()["camera"]["fps"])))
            time.sleep(1.0/fps)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

# -------- Import (YAML/JSON) --------
@APP.route("/import", methods=["GET"])
def import_form():
    cfg = load_cfg()
    with detector_lock:
        running = detector.is_running() if detector else False
        cam = detector.camera_desc if detector else "n/a"
    state = {"running": running, "camera": cam}
    devices = {"csi_available": csi_available(), "video_nodes": list_video_nodes()}
    return render_template("index.html", cfg=cfg, running=running, state=state, devices=devices, show_import=True)

@APP.route("/import", methods=["POST"])
def import_config():
    payload = request.form.get("payload", "").strip()
    if not payload:
        abort(400, "No payload")
    try:
        try:
            data = yaml.safe_load(payload)
        except Exception:
            data = json.loads(payload)
        if not isinstance(data, dict):
            abort(400, "Payload must be a dict")
        current = load_cfg()
        deep_merge(current, data)
        save_cfg(current)
        return redirect(url_for("index"))
    except Exception as e:
        return f"Import error: {e}", 400

# --------------- Main ---------------
if __name__ == "__main__":
    APP.run(host=os.getenv("AH_HOST","0.0.0.0"),
            port=int(os.getenv("AH_PORT","8080")),
            threaded=True)

