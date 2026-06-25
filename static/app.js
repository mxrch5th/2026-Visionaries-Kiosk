const video = document.getElementById('webcam');
const statusText = document.getElementById('status-text');
const probabilityText = document.getElementById('probability-text');
const menuTitle = document.getElementById('menu-title');
const menuContainer = document.getElementById('menu-container');

let currentMode = "normal"; 
let isAgeChecked = false; // 확실하게 얼굴 판별이 끝났는지 여부
let checkInterval = null;  // 반복을 제어할 인터벌 변수

const menus = {
    normal: [
        { name: "아메리카노", price: 3000, icon: "☕" },
        { name: "카페라떼", price: 4000, icon: "🥛" },
        { name: "아인슈페너", price: 5500, icon: "🧁" },
        { name: "바닐라라떼", price: 4500, icon: "🍯" }
    ],
    easy: [
        { name: "따뜻한 아메리카노 (커피)", price: 3000, icon: "☕" },
        { name: "달콤한 우유커피 (라떼)", price: 4000, icon: "🤎" },
        { name: "몸에 좋은 대추차", price: 5000, icon: "🍵" },
        { name: "시원하고 달달한 식혜", price: 4000, icon: "🍚" }
    ]
};

let cart = [];

// 1. 웹캠 켜기 및 스캔 루프 시작
navigator.mediaDevices.getUserMedia({ video: { width: 400, height: 300 } })
    .then((stream) => {
        video.srcObject = stream;
        statusText.innerText = "사용자를 인식하는 중입니다...";
        
        // 얼굴이 제대로 감지될 때까지 0.5초마다 계속 시도 (단, 성공하면 바로 멈춤)
        checkInterval = setInterval(scanUntilSuccess, 500);
    })
    .catch(err => {
        console.error("카메라 에러:", err);
        statusText.innerText = "카메라 연결 실패";
    });

// 2. 성공할 때까지만 스캔하는 함수
function scanUntilSuccess() {
    // 만약 다른 경로로 이미 판별이 완료되었다면 루프 종료
    if (isAgeChecked) {
        clearInterval(checkInterval);
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth || 400;
    canvas.height = video.videoHeight || 300;
    
    const ctx = canvas.getContext('2d');
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    const dataUrl = canvas.toDataURL('image/jpeg');

    // 백엔드 주소
    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: dataUrl })
    })
    .then(res => res.json())
    .then(data => {
        // 만약 백엔드에서 얼굴을 못 찾았다고 하면 고정하지 않고 다음 0.5초 뒤에 다시 찍음
        if (data.message === "얼굴 미감지" || data.prediction === undefined) {
            statusText.innerText = "카메라를 정면으로 바라봐주세요.";
            return;
        }

        // 얼굴이 제대로 감지되어 판별값이 넘어온 순간
        isAgeChecked = true; 
        clearInterval(checkInterval); // 0.5초마다 찍던 반복 타이머를 완전히 정지
        
        let prob = data.percent !== undefined ? data.percent : (data.confidence * 100 || 0);
        probabilityText.innerText = `50대 이상 가능성: ${prob.toFixed(1)}%`;

        if (data.prediction === "over50") {
            currentMode = "easy";
            statusText.innerText = "쉬운모드 활성화";
        } else {
            currentMode = "normal";
            statusText.innerText = "일반모드 활성화";
        }
        renderMenu();
    })
    .catch(err => console.error("스캔 중 에러:", err));
}

// 3. 메뉴판 그리기
function renderMenu() {
    menuContainer.innerHTML = "";
    
    if (currentMode === "easy") {
        menuTitle.innerHTML = "☀️ 쉬운모드 메뉴판";
        menuContainer.className = "menu-grid easy-mode";
    } else {
        menuTitle.innerHTML = "☕ 추천 메뉴판";
        menuContainer.className = "menu-grid";
    }

    menus[currentMode].forEach(item => {
        const itemBtn = document.createElement('button');
        itemBtn.className = 'menu-card-btn'; 
        itemBtn.innerHTML = `
            <div class="menu-icon">${item.icon}</div>
            <div class="menu-info">
                <div class="menu-name">${item.name}</div>
                <div class="menu-price">${item.price.toLocaleString()}원</div>
            </div>
        `;
        itemBtn.onclick = () => addToCart(item);
        menuContainer.appendChild(itemBtn);
    });
}

function addToCart(item) {
    const exist = cart.find(c => c.name === item.name);
    if (exist) {
        exist.count++;
    } else {
        cart.push({ ...item, count: 1 });
    }
    renderCart();
}

function renderCart() {
    const cartItems = document.getElementById('cart-items');
    const totalPrice = document.getElementById('total-price');
    cartItems.innerHTML = "";
    
    let total = 0;
    cart.forEach(item => {
        total += item.price * item.count;
        const div = document.createElement('div');
        div.className = 'cart-item';
        div.innerHTML = `<span>${item.name} x ${item.count}</span><span>${(item.price * item.count).toLocaleString()}원</span>`;
        cartItems.appendChild(div);
    });
    totalPrice.innerText = `총 금액: ${total.toLocaleString()}원`;
}

// 결제 완료 시 리셋하고 다시 다음 손님 스캔 루프 돌리기
function clearCart() {
    alert('결제가 완료되었습니다!');
    cart = [];
    isAgeChecked = false; 
    currentMode = "normal"; 
    statusText.innerText = "사용자를 인식하는 중입니다...";
    probabilityText.innerText = "50대 이상 가능성: 0.0%";
    renderMenu();
    renderCart();

    // 기존 인터벌 끄고 새로 스캔 시작
    clearInterval(checkInterval);
    checkInterval = setInterval(scanUntilSuccess, 500);
}

window.clearCart = clearCart;
renderMenu();
