const kiosk = document.querySelector(".kiosk");
const menuGrid = document.querySelector("#menuGrid");
const cartItems = document.querySelector("#cartItems");
const totalPrice = document.querySelector("#totalPrice");
const modePill = document.querySelector("#modePill");
const camera = document.querySelector("#camera");
const snapshot = document.querySelector("#snapshot");
const cameraStatus = document.querySelector("#cameraStatus");

const menu = {
  coffee: [
    { id: "americano", name: "아메리카노", desc: "깔끔한 기본 커피", price: 3500, icon: "☕" },
    { id: "latte", name: "카페라떼", desc: "부드러운 우유 커피", price: 4500, icon: "🥛" },
    { id: "vanilla", name: "바닐라라떼", desc: "달콤한 바닐라 향", price: 5000, icon: "🍯" },
    { id: "mocha", name: "카페모카", desc: "초콜릿이 들어간 커피", price: 5200, icon: "🍫" },
    { id: "coldbrew", name: "콜드브루", desc: "천천히 내린 시원한 커피", price: 4800, icon: "🧊" },
    { id: "espresso", name: "에스프레소", desc: "진한 한 잔", price: 3200, icon: "⚫" },
  ],
  nonCoffee: [
    { id: "greentea", name: "녹차라떼", desc: "고소한 녹차와 우유", price: 5000, icon: "🍵" },
    { id: "choco", name: "초코라떼", desc: "진한 초코 음료", price: 4800, icon: "🍫" },
    { id: "ade", name: "레몬에이드", desc: "상큼한 탄산 음료", price: 5200, icon: "🍋" },
    { id: "tea", name: "캐모마일 티", desc: "편하게 마시는 따뜻한 차", price: 4200, icon: "🌼" },
  ],
  dessert: [
    { id: "cake", name: "치즈케이크", desc: "진하고 부드러운 케이크", price: 6200, icon: "🍰" },
    { id: "cookie", name: "초코쿠키", desc: "바삭한 초코 쿠키", price: 2800, icon: "🍪" },
    { id: "bagel", name: "플레인 베이글", desc: "든든한 기본 베이글", price: 3900, icon: "🥯" },
    { id: "sandwich", name: "햄치즈 샌드위치", desc: "간단한 식사 메뉴", price: 6500, icon: "🥪" },
  ],
};

let activeCategory = "coffee";
let cart = [];
let lastMode = "normal";

function formatWon(value) {
  return `${value.toLocaleString("ko-KR")}원`;
}

function setMode(mode) {
  if (lastMode === mode) return;
  lastMode = mode;
  kiosk.dataset.mode = mode;
  modePill.textContent = mode === "easy" ? "쉬운모드" : "일반 화면";
}

function renderMenu() {
  menuGrid.innerHTML = menu[activeCategory]
    .map(
      (item) => `
        <article class="menu-card">
          <div class="menu-emoji" aria-hidden="true">${item.icon}</div>
          <div class="menu-name">${item.name}</div>
          <div class="menu-desc">${item.desc}</div>
          <div class="menu-bottom">
            <span class="price">${formatWon(item.price)}</span>
            <button class="add-button" data-id="${item.id}">담기</button>
          </div>
        </article>
      `
    )
    .join("");
}

function findItem(id) {
  return Object.values(menu)
    .flat()
    .find((item) => item.id === id);
}

function addToCart(id) {
  const item = findItem(id);
  const existing = cart.find((cartItem) => cartItem.id === id);
  if (existing) {
    existing.quantity += 1;
  } else {
    cart.push({ ...item, quantity: 1 });
  }
  renderCart();
}

function renderCart() {
  if (cart.length === 0) {
    cartItems.innerHTML = '<div class="cart-empty">담긴 메뉴가 없습니다</div>';
    totalPrice.textContent = "0원";
    return;
  }

  cartItems.innerHTML = cart
    .map(
      (item) => `
        <div class="cart-item">
          <div>
            <strong>${item.name}</strong>
            <div class="quantity">${item.quantity}개</div>
          </div>
          <span>${formatWon(item.price * item.quantity)}</span>
        </div>
      `
    )
    .join("");

  const total = cart.reduce((sum, item) => sum + item.price * item.quantity, 0);
  totalPrice.textContent = formatWon(total);
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
    camera.srcObject = stream;
    cameraStatus.textContent = "얼굴 인식 대기 중";
    setInterval(analyzeFrame, 3200);
  } catch (error) {
    cameraStatus.textContent = "카메라 권한이 필요합니다";
  }
}

async function analyzeFrame() {
  if (!camera.videoWidth) return;

  const context = snapshot.getContext("2d");
  context.drawImage(camera, 0, 0, snapshot.width, snapshot.height);
  const image = snapshot.toDataURL("image/jpeg", 0.75);

  try {
    const response = await fetch("/api/analyze-frame", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image }),
    });
    const result = await response.json();
    setMode(result.mode);

    const percent = Math.round(result.confidence * 100);
    const modelText = result.modelReady ? "모델 사용 중" : "모의 판별";
    const faceText = result.faceDetected ? "얼굴 감지" : "얼굴 미감지";
    cameraStatus.textContent = `${faceText} · 50대 이상 가능성 ${percent}% · ${modelText}`;
  } catch (error) {
    cameraStatus.textContent = "판별 서버 연결 확인 필요";
  }
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelector(".tab.active").classList.remove("active");
    tab.classList.add("active");
    activeCategory = tab.dataset.category;
    renderMenu();
  });
});

menuGrid.addEventListener("click", (event) => {
  const button = event.target.closest(".add-button");
  if (button) addToCart(button.dataset.id);
});

document.querySelector("#clearCart").addEventListener("click", () => {
  cart = [];
  renderCart();
});

document.querySelector("#payButton").addEventListener("click", () => {
  if (cart.length === 0) return;
  alert("주문이 접수되었습니다.");
  cart = [];
  renderCart();
});

renderMenu();
renderCart();
startCamera();
