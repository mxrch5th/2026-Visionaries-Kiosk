"""
키오스크 연령 인식 UI 최적화 시스템
팀: 비저너리즈 (Visionareis)

기능:
1. 웹캠으로 얼굴 실시간 감지
2. MobileNetV2 기반 연령 분류 (50대 이상 / 미만)
3. 고령층 감지 시 → 큰 글씨 + 쉬운 한국어 모드로 자동 전환
4. 일반 모드 / 고령자 모드 수동 토글 지원
"""

import sys, os, threading, time
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np

# ═══════════════════════════════════════════════════════
#  설정
# ═══════════════════════════════════════════════════════
MODEL_PATH        = os.path.join(os.path.dirname(__file__), "best_model_v3_final.pth")
DETECT_INTERVAL   = 2.5     # 연령 판단 간격 (초)
SENIOR_THRESHOLD  = 0.60    # senior 확률 임계값
SENIOR_PERSIST    = 15      # 고령자 모드 유지 시간 (초)
SENIOR_CONFIRM    = 2       # 연속 N회 감지돼야 전환
YOUNG_CONFIRM     = 5       # 연속 N회 young이어야 일반 모드 복귀

# ═══════════════════════════════════════════════════════
#  메뉴 데이터
# ═══════════════════════════════════════════════════════
MENU_DATA = [
    {"id":1,"name":"아메리카노","easy_name":"아메리카노\n(블랙커피)","price":4500,
     "emoji":"☕","category":"커피","desc":"Espresso & Water",
     "easy_desc":"진한 커피에\n물을 섞은 음료예요"},
    {"id":2,"name":"카페라떼","easy_name":"카페라떼\n(커피우유)","price":5000,
     "emoji":"🥛","category":"커피","desc":"Espresso & Steamed Milk",
     "easy_desc":"커피와 따뜻한\n우유를 섞은 음료예요"},
    {"id":3,"name":"카푸치노","easy_name":"카푸치노\n(거품커피)","price":5000,
     "emoji":"☕","category":"커피","desc":"Espresso & Foam",
     "easy_desc":"커피 위에 우유\n거품을 올린 음료예요"},
    {"id":4,"name":"그린티 라떼","easy_name":"녹차 우유","price":5500,
     "emoji":"🍵","category":"티·에이드","desc":"Matcha & Steamed Milk",
     "easy_desc":"녹차 가루를 우유에\n섞은 음료예요"},
    {"id":5,"name":"레몬 에이드","easy_name":"레몬\n탄산음료","price":5500,
     "emoji":"🍋","category":"티·에이드","desc":"Fresh Lemon & Sparkling",
     "easy_desc":"레몬과 탄산수로\n만든 시원한 음료예요"},
    {"id":6,"name":"초코 스무디","easy_name":"초콜릿\n음료","price":6000,
     "emoji":"🍫","category":"스무디","desc":"Belgian Chocolate Blend",
     "easy_desc":"초콜릿으로 만든\n달콤한 음료예요"},
    {"id":7,"name":"딸기 스무디","easy_name":"딸기 음료","price":6500,
     "emoji":"🍓","category":"스무디","desc":"Fresh Strawberry Blend",
     "easy_desc":"신선한 딸기로\n만든 음료예요"},
    {"id":8,"name":"치즈케이크","easy_name":"치즈케이크","price":6500,
     "emoji":"🍰","category":"디저트","desc":"New York Style",
     "easy_desc":"부드러운 치즈로\n만든 케이크예요"},
]

# ═══════════════════════════════════════════════════════
#  테마 토큰
# ═══════════════════════════════════════════════════════
N = {  # Normal
    "bg":          "#F7F8FA",
    "sidebar_bg":  "#0F172A",
    "sidebar_fg":  "#94A3B8",
    "sidebar_acc": "#38BDF8",
    "card_bg":     "#FFFFFF",
    "card_hover":  "#F0F9FF",
    "card_border": "#E2E8F0",
    "primary":     "#0EA5E9",
    "primary_dk":  "#0284C7",
    "success":     "#10B981",
    "danger":      "#EF4444",
    "text":        "#0F172A",
    "text2":       "#475569",
    "text3":       "#94A3B8",
    "price":       "#0EA5E9",
    "tag_bg":      "#EFF6FF",
    "tag_fg":      "#3B82F6",
    "status_bg":   "#1E293B",
    "status_fg":   "#94A3B8",
    # fonts
    "fh": 15,   # header
    "ft": 12,   # card title
    "fb": 10,   # body / desc
    "fp": 12,   # price
    "fn": 10,   # btn
    "cols": 4,
    "card_w": 190,
    "card_h": 200,
    "btn_h": 30,
}

S = {  # Senior
    "bg":          "#FFFBF0",
    "sidebar_bg":  "#1A3A2A",
    "sidebar_fg":  "#A7C9B5",
    "sidebar_acc": "#5EE896",
    "card_bg":     "#FFFFFF",
    "card_hover":  "#F0FFF4",
    "card_border": "#86EFAC",
    "primary":     "#16A34A",
    "primary_dk":  "#15803D",
    "success":     "#16A34A",
    "danger":      "#DC2626",
    "text":        "#111827",
    "text2":       "#1F2937",
    "text3":       "#374151",
    "price":       "#92400E",
    "tag_bg":      "#ECFDF5",
    "tag_fg":      "#065F46",
    "status_bg":   "#14532D",
    "status_fg":   "#A7C9B5",
    # fonts
    "fh": 22,
    "ft": 19,
    "fb": 16,
    "fp": 20,
    "fn": 17,
    "cols": 2,
    "card_w": 280,
    "card_h": 270,
    "btn_h": 52,
}

FONT = "Malgun Gothic"

# ═══════════════════════════════════════════════════════
#  AI 모델
# ═══════════════════════════════════════════════════════
def load_model():
    device = torch.device("cpu")
    model  = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    model = model.to(device)
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.eval()
            print(f"✅ 모델 로드: {MODEL_PATH}")
            return model, device
        except Exception as e:
            print(f"⚠️  로드 실패: {e}")
    else:
        print("⚠️  모델 없음 → 데모 모드")
    return None, device

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def predict_age(model, device, frame_bgr):
    if model is None:
        p = np.random.random()
        lbl = "senior" if p > 0.5 else "young"
        return lbl, p if lbl == "senior" else 1-p
    img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    t   = TRANSFORM(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out   = model(t)
        probs = torch.softmax(out, dim=1)[0]
        pred  = probs.argmax().item()
    lbl = "senior" if pred == 1 else "young"
    return lbl, probs[pred].item()

def get_face_cascade():
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(gray, cascade):
    return cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80,80))

# ═══════════════════════════════════════════════════════
#  둥근 모서리 이미지 헬퍼 (PIL)
# ═══════════════════════════════════════════════════════
def rounded_rect_image(w, h, r, color):
    img  = Image.new("RGBA", (w, h), (0,0,0,0))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle([0,0,w-1,h-1], radius=r, fill=color)
    return ImageTk.PhotoImage(img)

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2],16) for i in (0,2,4))

# ═══════════════════════════════════════════════════════
#  메인 앱
# ═══════════════════════════════════════════════════════
class KioskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("스마트 카페 키오스크  ·  비저너리즈")
        self.root.geometry("1280x820")
        self.root.minsize(1100, 720)
        self.root.configure(bg="#0F172A")

        self.is_senior   = False
        self.last_senior = 0.0
        self.cart        = {}
        self.cam_running = False
        self.cap         = None
        self.model, self.device = load_model()
        self.face_cascade = get_face_cascade()

        self.det_status  = tk.StringVar(value="  📷  카메라 연결 중...")
        self.det_prob    = tk.StringVar(value="")

        self._img_refs   = []        # PhotoImage GC 방지
        self._cam_label  = None      # 미리보기 Label 위젯
        self._last_faces = []        # 최근 감지된 얼굴 박스
        self._last_lbl   = "young"  # 최근 판정 결과
        self._senior_cnt = 0         # 연속 senior 감지 횟수
        self._young_cnt  = 0         # 연속 young 감지 횟수

        self._build()
        # 앱 시작 즉시 카메라 자동 켜기
        self.root.after(300, self._start_cam)

    # ────────────────────────────────────────────────────
    #  테마 헬퍼
    # ────────────────────────────────────────────────────
    def t(self):
        return S if self.is_senior else N

    # ────────────────────────────────────────────────────
    #  전체 빌드 / 리빌드
    # ────────────────────────────────────────────────────
    def _build(self):
        self._img_refs.clear()
        for w in self.root.winfo_children():
            w.destroy()

        T = self.t()
        self.root.configure(bg=T["sidebar_bg"])

        # ── 최상단 Status Bar ───────────────────────────
        self._build_statusbar(T)

        # ── 메인 콘텐츠 (사이드바 + 메뉴영역 + 카트) ──
        main = tk.Frame(self.root, bg=T["bg"])
        main.pack(fill="both", expand=True)

        self._build_sidebar(main, T)
        self._build_menu_area(main, T)
        self._build_cart(main, T)

    def _build_statusbar(self, T):
        bar = tk.Frame(self.root, bg=T["status_bg"], height=38)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        # 감지 상태 텍스트
        self.status_lbl = tk.Label(
            bar, textvariable=self.det_status,
            bg=T["status_bg"], fg=T["status_fg"],
            font=(FONT, 9))
        self.status_lbl.pack(side="left", padx=12)

        self.prob_lbl = tk.Label(
            bar, textvariable=self.det_prob,
            bg=T["status_bg"], fg=T["sidebar_acc"],
            font=(FONT, 9, "bold"))
        self.prob_lbl.pack(side="left")

        # 오른쪽 컨트롤 (모드 전환 버튼만)
        ctrl = tk.Frame(bar, bg=T["status_bg"])
        ctrl.pack(side="right", padx=10)

        mode_txt = "일반 모드로 전환" if self.is_senior else "👴  큰글씨 모드"
        mode_bg  = "#374151" if self.is_senior else "#166534"
        self.mode_btn = tk.Button(
            ctrl, text=mode_txt,
            bg=mode_bg, fg="#DCFCE7" if not self.is_senior else "#D1D5DB",
            font=(FONT, 9, "bold"),
            relief="flat", padx=10, pady=4,
            cursor="hand2", bd=0,
            activebackground="#15803D", activeforeground="white",
            command=self._toggle_manual)
        self.mode_btn.pack(side="left", padx=4)

    # ── 사이드바 ─────────────────────────────────────────
    def _build_sidebar(self, parent, T):
        sb = tk.Frame(parent, bg=T["sidebar_bg"], width=200)
        sb.pack(side="left", fill="y")
        sb.pack_propagate(False)

        # 로고
        logo_frame = tk.Frame(sb, bg=T["sidebar_bg"])
        logo_frame.pack(fill="x", pady=(20,0), padx=16)

        tk.Label(logo_frame, text="☕", bg=T["sidebar_bg"],
                 font=("Segoe UI Emoji", 28)).pack(anchor="w")
        tk.Label(logo_frame, text="CAFÉ",
                 bg=T["sidebar_bg"], fg="white",
                 font=(FONT, 18, "bold")).pack(anchor="w")
        tk.Label(logo_frame, text="VISIONAREIS",
                 bg=T["sidebar_bg"], fg=T["sidebar_fg"],
                 font=(FONT, 8)).pack(anchor="w")

        # 구분선
        tk.Frame(sb, bg=T["sidebar_fg"], height=1).pack(
            fill="x", padx=16, pady=20)

        # 카테고리 목록
        cats = list(dict.fromkeys(m["category"] for m in MENU_DATA))
        cat_icons = {"커피":"☕", "티·에이드":"🍵", "스무디":"🥤", "디저트":"🍰"}

        for cat in cats:
            icon = cat_icons.get(cat, "•")
            row = tk.Frame(sb, bg=T["sidebar_bg"], cursor="hand2")
            row.pack(fill="x", padx=8, pady=1)

            lbl = tk.Label(row,
                text=f"  {icon}  {cat}",
                bg=T["sidebar_bg"], fg=T["sidebar_fg"],
                font=(FONT, T["fb"]),
                anchor="w", padx=4, pady=8)
            lbl.pack(fill="x")
            lbl.bind("<Enter>", lambda e, l=lbl: l.config(
                fg=T["sidebar_acc"], bg="#1E293B" if not self.is_senior else "#1A4530"))
            lbl.bind("<Leave>", lambda e, l=lbl: l.config(
                fg=T["sidebar_fg"], bg=T["sidebar_bg"]))

        # 하단 모드 뱃지
        badge_frame = tk.Frame(sb, bg=T["sidebar_bg"])
        badge_frame.pack(side="bottom", fill="x", padx=12, pady=(0, 10))

        if self.is_senior:
            tk.Label(badge_frame,
                text="👴  큰글씨 모드 활성",
                bg="#14532D", fg="#BBF7D0",
                font=(FONT, 9, "bold"),
                padx=8, pady=6).pack(fill="x")
        else:
            tk.Label(badge_frame,
                text="👤  일반 모드",
                bg="#1E3A5F", fg="#93C5FD",
                font=(FONT, 9),
                padx=8, pady=6).pack(fill="x")

        # ── 카메라 미리보기 패널 ──────────────────────────
        cam_frame = tk.Frame(sb, bg=T["sidebar_bg"])
        cam_frame.pack(side="bottom", fill="x", padx=12, pady=(0, 6))

        tk.Label(cam_frame,
            text="🎥  얼굴 인식 중",
            bg=T["sidebar_bg"], fg=T["sidebar_fg"],
            font=(FONT, 8)).pack(anchor="w", pady=(0, 3))

        # 미리보기 Label (168x126)
        self._cam_label = tk.Label(
            cam_frame, bg="#000000",
            relief="flat")
        self._cam_label.pack(fill="x")

    # ── 메뉴 영역 ─────────────────────────────────────────
    def _build_menu_area(self, parent, T):
        area = tk.Frame(parent, bg=T["bg"])
        area.pack(side="left", fill="both", expand=True)

        # 헤더
        hdr = tk.Frame(area, bg=T["bg"])
        hdr.pack(fill="x", padx=24, pady=(18,0))

        if self.is_senior:
            tk.Label(hdr, text="메뉴를 골라주세요",
                     bg=T["bg"], fg=T["text"],
                     font=(FONT, T["fh"]+2, "bold")).pack(side="left")
        else:
            tk.Label(hdr, text="Menu",
                     bg=T["bg"], fg=T["text"],
                     font=(FONT, T["fh"]+2, "bold")).pack(side="left")
            tk.Label(hdr, text=f"  {len(MENU_DATA)}가지 음료 · 디저트",
                     bg=T["bg"], fg=T["text3"],
                     font=(FONT, T["fb"])).pack(side="left", pady=(6,0))

        # 스크롤 캔버스
        canvas = tk.Canvas(area, bg=T["bg"], highlightthickness=0)
        vsb    = ttk.Scrollbar(area, orient="vertical", command=canvas.yview)
        inner  = tk.Frame(canvas, bg=T["bg"])

        inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(16,0), pady=12)
        vsb.pack(side="right", fill="y", pady=12)

        canvas.bind_all("<MouseWheel>",
            lambda e: canvas.yview_scroll(-1*(e.delta//120), "units"))

        self._fill_menu_grid(inner, T)

    def _fill_menu_grid(self, parent, T):
        from collections import OrderedDict
        cats = OrderedDict()
        for m in MENU_DATA:
            cats.setdefault(m["category"], []).append(m)

        cols  = T["cols"]
        r_off = 0

        for cat, items in cats.items():
            # 카테고리 헤더
            cat_icons = {"커피":"☕", "티·에이드":"🍵", "스무디":"🥤", "디저트":"🍰"}
            icon = cat_icons.get(cat,"")
            cat_lbl = tk.Label(parent,
                text=f"  {icon}  {cat}",
                bg=T["bg"], fg=T["text2"],
                font=(FONT, T["fb"]+1, "bold"),
                anchor="w")
            cat_lbl.grid(row=r_off, column=0, columnspan=cols,
                         sticky="ew", padx=8, pady=(16,4))
            r_off += 1

            for idx, item in enumerate(items):
                c = idx % cols
                r = r_off + idx // cols
                card = self._make_card(item, T, parent)
                card.grid(row=r, column=c,
                          padx=8, pady=8, sticky="nsew")
                parent.columnconfigure(c, weight=1)

            r_off += (len(items)-1)//cols + 1

    def _make_card(self, item, T, parent):
        card = tk.Frame(parent,
            bg=T["card_bg"],
            highlightthickness=1,
            highlightbackground=T["card_border"],
            cursor="hand2")

        name = item["easy_name"] if self.is_senior else item["name"]
        desc = item["easy_desc"] if self.is_senior else item["desc"]

        # 호버 효과
        def on_enter(e):
            card.config(highlightbackground=T["primary"],
                        highlightthickness=2)
        def on_leave(e):
            card.config(highlightbackground=T["card_border"],
                        highlightthickness=1)
        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)

        # 이모지
        emoji_lbl = tk.Label(card, text=item["emoji"],
            bg=T["card_bg"],
            font=("Segoe UI Emoji",
                  32 if self.is_senior else 24))
        emoji_lbl.pack(pady=(16 if self.is_senior else 12, 4))

        # 상품명
        tk.Label(card, text=name,
            bg=T["card_bg"], fg=T["text"],
            font=(FONT, T["ft"], "bold"),
            wraplength=T["card_w"]-20,
            justify="center").pack(padx=10)

        # 설명
        tk.Label(card, text=desc,
            bg=T["card_bg"], fg=T["text3" if not self.is_senior else "text2"],
            font=(FONT, T["fb"]),
            wraplength=T["card_w"]-20,
            justify="center").pack(padx=10, pady=(3,0))

        # 가격
        price_txt = (f"{item['price']:,}원" if self.is_senior
                     else f"₩ {item['price']:,}")
        tk.Label(card, text=price_txt,
            bg=T["card_bg"], fg=T["price"],
            font=(FONT, T["fp"], "bold")).pack(pady=(8,0))

        # 담기 버튼
        btn_txt  = "🛒  담기" if self.is_senior else "ADD"
        btn_font = (FONT, T["fn"], "bold")
        btn = tk.Button(card,
            text=btn_txt,
            bg=T["primary"], fg="white",
            font=btn_font,
            relief="flat", cursor="hand2",
            pady=8 if self.is_senior else 5,
            activebackground=T["primary_dk"],
            activeforeground="white",
            command=lambda iid=item["id"]: self._add(iid))
        btn.pack(fill="x", padx=12,
                 pady=(8 if self.is_senior else 6,
                       12 if self.is_senior else 8))

        return card

    # ── 카트 ────────────────────────────────────────────
    def _build_cart(self, parent, T):
        cart_w = 300 if self.is_senior else 260
        self.cart_frame = tk.Frame(parent,
            bg=T["card_bg"],
            width=cart_w,
            highlightthickness=1,
            highlightbackground=T["card_border"])
        self.cart_frame.pack(side="right", fill="y",
                             padx=(0,0), pady=0)
        self.cart_frame.pack_propagate(False)

        # 카트 헤더
        ch = tk.Frame(self.cart_frame, bg=T["sidebar_bg"], height=52)
        ch.pack(fill="x")
        ch.pack_propagate(False)

        title = "주문 목록" if self.is_senior else "ORDER"
        tk.Label(ch, text=f"🛒  {title}",
            bg=T["sidebar_bg"], fg="white",
            font=(FONT, T["fb"]+3, "bold")).pack(
            side="left", padx=16, pady=14)

        # 카트 내용 (스크롤)
        self.cart_inner = tk.Frame(self.cart_frame, bg=T["card_bg"])
        self.cart_inner.pack(fill="both", expand=True, padx=10, pady=8)

        # 하단 고정 영역
        bottom = tk.Frame(self.cart_frame, bg=T["card_bg"])
        bottom.pack(fill="x", side="bottom", padx=12, pady=(0,10))

        divider = tk.Frame(bottom, bg=T["card_border"], height=1)
        divider.pack(fill="x", pady=(0,10))

        total_row = tk.Frame(bottom, bg=T["card_bg"])
        total_row.pack(fill="x", pady=(0,8))

        total_lbl = "합  계" if self.is_senior else "Total"
        tk.Label(total_row, text=total_lbl,
            bg=T["card_bg"], fg=T["text2"],
            font=(FONT, T["fb"]+1)).pack(side="left")

        self.total_var = tk.StringVar(value="0원" if self.is_senior else "₩ 0")
        tk.Label(total_row,
            textvariable=self.total_var,
            bg=T["card_bg"], fg=T["text"],
            font=(FONT, T["fp"]+2, "bold")).pack(side="right")

        # 전체 삭제 버튼 (작게)
        tk.Button(bottom, text="전체 삭제",
            bg=T["card_bg"], fg=T["text3"],
            font=(FONT, 8), relief="flat",
            cursor="hand2",
            command=self._clear_cart).pack(anchor="e", pady=(0,4))

        # 결제 버튼
        pay_txt = "결  제  하  기 →" if self.is_senior else "결제하기  →"
        self.pay_btn = tk.Button(
            bottom, text=pay_txt,
            bg=T["primary"], fg="white",
            font=(FONT, T["fn"]+2, "bold"),
            relief="flat", pady=14 if self.is_senior else 11,
            cursor="hand2",
            activebackground=T["primary_dk"],
            activeforeground="white",
            command=self._checkout)
        self.pay_btn.pack(fill="x")

        self._refresh_cart()

    def _refresh_cart(self):
        T = self.t()
        for w in self.cart_inner.winfo_children():
            w.destroy()

        if not self.cart:
            empty = "담긴 메뉴가 없어요 😊" if self.is_senior else "장바구니가 비어 있습니다"
            tk.Label(self.cart_inner, text=empty,
                bg=T["card_bg"], fg=T["text3"],
                font=(FONT, T["fb"]),
                wraplength=220, justify="center").pack(pady=30)
            self.total_var.set("0원" if self.is_senior else "₩ 0")
            return

        total = 0
        for iid, qty in self.cart.items():
            item    = next(x for x in MENU_DATA if x["id"]==iid)
            sub     = item["price"] * qty
            total  += sub
            name    = item["easy_name"].replace("\n"," ") if self.is_senior else item["name"]

            row = tk.Frame(self.cart_inner,
                bg=T["card_bg"],
                highlightthickness=1,
                highlightbackground=T["card_border"])
            row.pack(fill="x", pady=3)

            left = tk.Frame(row, bg=T["card_bg"])
            left.pack(side="left", fill="x", expand=True, padx=(8,0), pady=6)

            tk.Label(left, text=f"{item['emoji']} {name}",
                bg=T["card_bg"], fg=T["text"],
                font=(FONT, T["fb"]+1, "bold"),
                anchor="w", wraplength=140).pack(anchor="w")
            tk.Label(left, text=f"{sub:,}원",
                bg=T["card_bg"], fg=T["price"],
                font=(FONT, T["fb"]),
                anchor="w").pack(anchor="w")

            # 수량 조절
            qctrl = tk.Frame(row, bg=T["card_bg"])
            qctrl.pack(side="right", padx=8, pady=6)

            btn_kw = dict(relief="flat", cursor="hand2",
                          font=(FONT, T["fn"], "bold"),
                          width=2 if not self.is_senior else 3,
                          pady=2 if not self.is_senior else 5)

            tk.Button(qctrl, text="−",
                bg=T["danger"], fg="white",
                activebackground="#B91C1C",
                command=lambda i=iid: self._qty(i,-1),
                **btn_kw).pack(side="left")

            tk.Label(qctrl, text=f"  {qty}  ",
                bg=T["card_bg"], fg=T["text"],
                font=(FONT, T["fb"]+1, "bold")).pack(side="left")

            tk.Button(qctrl, text="+",
                bg=T["success"], fg="white",
                activebackground="#047857",
                command=lambda i=iid: self._qty(i, 1),
                **btn_kw).pack(side="left")

        total_str = f"{total:,}원" if self.is_senior else f"₩ {total:,}"
        self.total_var.set(total_str)

    def _add(self, iid):
        self.cart[iid] = self.cart.get(iid, 0) + 1
        self._refresh_cart()

    def _qty(self, iid, delta):
        nq = self.cart.get(iid, 0) + delta
        if nq <= 0:
            self.cart.pop(iid, None)
        else:
            self.cart[iid] = nq
        self._refresh_cart()

    def _clear_cart(self):
        self.cart.clear()
        self._refresh_cart()

    def _checkout(self):
        T = self.t()
        if not self.cart:
            msg = "담긴 메뉴가 없어요.\n메뉴를 먼저 골라주세요." if self.is_senior \
                  else "장바구니가 비어 있습니다."
            messagebox.showinfo("알림", msg)
            return

        total = sum(
            next(x for x in MENU_DATA if x["id"]==iid)["price"] * qty
            for iid, qty in self.cart.items())

        lines = []
        for iid, qty in self.cart.items():
            item = next(x for x in MENU_DATA if x["id"]==iid)
            n    = item["easy_name"].replace("\n"," ") if self.is_senior else item["name"]
            lines.append(f"  {item['emoji']}  {n}  ×{qty}  →  {item['price']*qty:,}원")

        body = "\n".join(lines) + f"\n\n합계:  {total:,}원"
        q = "주문하시겠어요?" if self.is_senior else "주문을 확정하시겠습니까?"

        if messagebox.askyesno(q, body):
            self.cart.clear()
            self._refresh_cart()
            done = ("주문이 완료되었습니다! 😊\n잠시 후 준비해 드릴게요."
                    if self.is_senior else "주문이 완료되었습니다. 감사합니다 😊")
            messagebox.showinfo("주문 완료", done)

    # ────────────────────────────────────────────────────
    #  모드 전환
    # ────────────────────────────────────────────────────
    def _set_mode(self, senior: bool, reason=""):
        if senior == self.is_senior:
            return
        self.is_senior = senior
        print(f"🔄  {'고령자 큰글씨' if senior else '일반'} 모드  ({reason})")
        self._build()

    def _toggle_manual(self):
        self._set_mode(not self.is_senior, "수동")

    # ────────────────────────────────────────────────────
    #  카메라
    # ────────────────────────────────────────────────────
    def _start_cam(self):
        if self.cam_running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.det_status.set("  ❌  카메라를 열 수 없습니다")
            self.cap = None
            return
        self.cam_running = True
        threading.Thread(target=self._cam_loop, daemon=True).start()

    def _stop_cam(self):
        self.cam_running = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def _cam_loop(self):
        last_detect = 0.0
        while self.cam_running:
            if not self.cap or not self.cap.isOpened():
                break
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            now = time.time()

            # AI 판정은 DETECT_INTERVAL 마다
            if now - last_detect >= DETECT_INTERVAL:
                last_detect = now
                self._detect(frame)

            # 매 프레임: 미리보기 업데이트
            self._update_preview(frame)

            # 카메라가 없거나 얼굴 미감지 상태가 오래되면 일반 모드 복귀
            if self.is_senior and (now - self.last_senior) > SENIOR_PERSIST:
                self.root.after(0, lambda: self._set_mode(False, "타임아웃"))
            time.sleep(0.03)

    def _update_preview(self, frame):
        """매 프레임 얼굴 박스를 그려서 사이드바 미리보기에 표시"""
        if self._cam_label is None:
            return
        try:
            # 리사이즈: 168x126 (사이드바 너비 맞춤)
            PREV_W, PREV_H = 168, 126
            small = cv2.resize(frame, (PREV_W, PREV_H))

            # 얼굴 박스 그리기 (저장된 최근 결과 사용)
            if self._last_faces is not None and len(self._last_faces) > 0:
                # 원본 → 미리보기 비율
                orig_h, orig_w = frame.shape[:2]
                rx, ry = PREV_W / orig_w, PREV_H / orig_h
                # 박스 색: 실제 모드 기준 (AI 판정 매번 반영 X → 깜빡임 방지)
                color = (0, 140, 255) if self.is_senior else (0, 220, 80)
                # 전환 대기 중(카운트 쌓이는 중)이면 노란색으로 표시
                if self._senior_cnt > 0 and not self.is_senior:
                    color = (0, 200, 200)  # 노란색: 감지 중
                for (fx, fy, fw, fh) in self._last_faces:
                    x1 = int(fx * rx);  y1 = int(fy * ry)
                    x2 = int((fx+fw)*rx); y2 = int((fy+fh)*ry)
                    cv2.rectangle(small, (x1, y1), (x2, y2), color, 2)
                    # 텍스트: 카운트 표시
                    if self.is_senior:
                        label_txt = "50+"
                    elif self._senior_cnt > 0:
                        label_txt = f"?{self._senior_cnt}/{SENIOR_CONFIRM}"
                    else:
                        label_txt = "OK"
                    cv2.putText(small, label_txt, (x1, max(y1-4, 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            # BGR → RGB → PIL → PhotoImage
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            pil   = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil)

            # 메인 스레드에서 Label 업데이트
            self.root.after(0, lambda p=photo: self._set_preview(p))
        except Exception:
            pass

    def _set_preview(self, photo):
        """메인 스레드: Label에 이미지 적용 (GC 방지 포함)"""
        try:
            if self._cam_label and self._cam_label.winfo_exists():
                self._cam_label.config(image=photo)
                self._cam_label.image = photo   # GC 방지
        except Exception:
            pass

    def _detect(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray, self.face_cascade)

        if len(faces) == 0:
            self._last_faces = []
            self.root.after(0, lambda: self.det_status.set(
                "  👁️  얼굴을 찾고 있어요..."))
            self.root.after(0, lambda: self.det_prob.set(""))
            return

        self._last_faces = list(faces)
        x,y,w,h = sorted(faces, key=lambda f:f[2]*f[3])[-1]
        roi  = frame[max(0,y-20):y+h+20, max(0,x-20):x+w+20]
        lbl, prob = predict_age(self.model, self.device, roi)
        self._last_lbl = lbl

        pct  = f"{prob*100:.0f}%"
        icon = "🧓" if lbl=="senior" else "👤"
        name = "50대 이상" if lbl=="senior" else "일반"
        msg  = f"  {icon}  얼굴 감지  |  {name} ({pct})"
        col  = "#FCA5A5" if lbl=="senior" else "#86EFAC"

        self.root.after(0, lambda: self.det_status.set(msg))
        self.root.after(0, lambda: self.prob_lbl.config(fg=col))
        self.root.after(0, lambda: self.det_prob.set(""))

        if lbl == "senior" and prob >= SENIOR_THRESHOLD:
            self._senior_cnt += 1
            self._young_cnt   = 0
            # 연속 SENIOR_CONFIRM 회 이상 감지돼야 전환
            if self._senior_cnt >= SENIOR_CONFIRM and not self.is_senior:
                self.last_senior = time.time()
                self.root.after(0, lambda: self._set_mode(True, f"AI {pct}"))
            elif self.is_senior:
                # 이미 고령자 모드면 타이머 갱신
                self.last_senior = time.time()
        else:
            self._young_cnt  += 1
            self._senior_cnt  = 0
            # 연속 YOUNG_CONFIRM 회 이상 young이어야 일반 모드 복귀
            if self._young_cnt >= YOUNG_CONFIRM and self.is_senior:
                self.root.after(0, lambda: self._set_mode(False, "연속 young"))

    def on_close(self):
        self._stop_cam()
        self.root.destroy()


# ═══════════════════════════════════════════════════════
#  진입점
# ═══════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app  = KioskApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
