<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Bác Sĩ Sầu Riêng AI</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background-color: #1e2533; 
            color: #dde3ee; 
            margin: 0; 
            padding: 15px; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            -webkit-user-select: none; 
            user-select: none;
        }
        .container { width: 100%; max-width: 400px; }
        h1 { text-align: center; color: #f0c040; font-size: 22px; margin-bottom: 5px; }
        .subtitle { text-align: center; color: #8899bb; font-size: 12px; margin-top: 0; margin-bottom: 15px; }
        
        .status-box { 
            background: #111827; 
            border: 1px solid #333; 
            border-radius: 10px; 
            padding: 20px; 
            text-align: center; 
            margin-bottom: 20px; 
            min-height: 130px; 
            display: flex; 
            flex-direction: column; 
            justify-content: center; 
        }
        #result-text { font-size: 32px; font-weight: bold; margin: 0; transition: color 0.3s; }
        #conf-text { font-size: 14px; color: #f0c040; margin-top: 5px; }
        #rms-text { font-size: 12px; color: #8899bb; margin-top: 5px; }
        
        .btn-record { 
            width: 100%; 
            padding: 20px; 
            font-size: 18px; 
            font-weight: bold; 
            border: none; 
            border-radius: 10px; 
            cursor: pointer; 
            background-color: #1a7a3a; 
            color: white; 
            box-shadow: 0 4px 10px rgba(0,0,0,0.3); 
            transition: 0.2s; 
        }
        .btn-record:active { transform: scale(0.95); }
        .btn-record:disabled { background-color: #555; cursor: not-allowed; }
        .btn-record.recording { 
            background-color: #e74c3c; 
            animation: pulse 1s infinite; 
        }
        @keyframes pulse { 
            0% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0.7); } 
            70% { box-shadow: 0 0 0 20px rgba(231, 76, 60, 0); } 
            100% { box-shadow: 0 0 0 0 rgba(231, 76, 60, 0); } 
        }

        .settings { 
            background: #0c1a28; 
            border-radius: 10px; 
            padding: 15px; 
            margin-top: 20px; 
        }
        .settings h3 { 
            margin-top: 0; 
            color: #00d4ff; 
            font-size: 16px; 
            border-bottom: 1px solid #333; 
            padding-bottom: 5px; 
        }
        .slider-group { margin-bottom: 15px; }
        .slider-group:last-child { margin-bottom: 0; }
        .slider-label { 
            display: flex; 
            justify-content: space-between; 
            font-size: 13px; 
            margin-bottom: 5px; 
        }
        .slider-value { color: #f0c040; font-weight: bold; }
        input[type=range] { 
            width: 100%; 
            accent-color: #f0c040; 
            height: 6px;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>🍈 BÁC SĨ SẦU RIÊNG AI</h1>
    <p class="subtitle">H5 Hybrid TFLite Model</p>
    
    <div class="status-box">
        <div id="result-text">Sẵn sàng đo</div>
        <div id="conf-text"></div>
        <div id="rms-text">RMS: --</div>
    </div>

    <button id="btn-record" class="btn-record">🎙️ ẤN VÀO ĐỂ GÕ SẦU RIÊNG</button>

    <div class="settings">
        <h3>⚙️ Cài đặt thông số (Dành cho chuyên viên)</h3>
        
        <div class="slider-group">
            <div class="slider-label">
                <span>Nền nhiễu (RMS Threshold)</span>
                <span id="val-rms" class="slider-value">0.0025</span>
            </div>
            <input type="range" id="slider-rms" min="0.0005" max="0.01" step="0.0001" value="0.0025">
        </div>

        <div class="slider-group">
            <div class="slider-label">
                <span>Ngưỡng Đủ Tuổi (%)</span>
                <span id="val-dt" class="slider-value">50%</span>
            </div>
            <input type="range" id="slider-dt" min="10" max="90" step="1" value="50">
        </div>

        <div class="slider-group">
            <div class="slider-label">
                <span>Ngưỡng Xanh (%)</span>
                <span id="val-xn" class="slider-value">35%</span>
            </div>
            <input type="range" id="slider-xn" min="10" max="90" step="1" value="35">
        </div>
    </div>
</div>

<script>
    // 🛑🛑🛑 THAY ĐỔI ĐỊA CHỈ DÒNG DƯỚI ĐÂY THÀNH LINK CỦA BẠN TRÊN HUGGING FACE 🛑🛑🛑
    // Ví dụ: https://nguyenvana-durian-ai.hf.space/predict
    const API_URL = "https://THAY-LINK-CUA-BAN-O-DAY.hf.space/predict"; 

    const btnRecord = document.getElementById('btn-record');
    const resultText = document.getElementById('result-text');
    const confText = document.getElementById('conf-text');
    const rmsText = document.getElementById('rms-text');

    const sliderRms = document.getElementById('slider-rms');
    const sliderDt = document.getElementById('slider-dt');
    const sliderXn = document.getElementById('slider-xn');

    // Cập nhật số liệu hiển thị khi kéo thanh trượt
    sliderRms.oninput = (e) => document.getElementById('val-rms').innerText = e.target.value;
    sliderDt.oninput = (e) => document.getElementById('val-dt').innerText = e.target.value + "%";
    sliderXn.oninput = (e) => document.getElementById('val-xn').innerText = e.target.value + "%";

    let mediaRecorder;
    let audioChunks = [];

    // Bắt đầu thu âm khi nhấn nút
    btnRecord.onclick = async () => {
        if(mediaRecorder && mediaRecorder.state === "recording") return;

        try {
            // Xin quyền sử dụng Microphone
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = event => { 
                audioChunks.push(event.data); 
            };
            
            // Khi dừng thu âm (sau 1.2 giây) -> Gửi đi
            mediaRecorder.onstop = async () => {
                btnRecord.classList.remove('recording');
                btnRecord.innerText = "🎙️ ẤN VÀO ĐỂ GÕ SẦU RIÊNG";
                btnRecord.disabled = false;
                resultText.innerText = "⏳ Đang phân tích AI...";
                resultText.style.color = "#dde3ee";
                
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                await sendToServer(audioBlob);
                
                // Tắt mic sau khi gửi xong để tiết kiệm pin
                stream.getTracks().forEach(track => track.stop()); 
            };

            // Bắt đầu ghi âm
            mediaRecorder.start();
            btnRecord.classList.add('recording');
            btnRecord.innerText = "🔴 ĐANG NGHE... (1.2s)";
            btnRecord.disabled = true;
            
            // Tự động dừng thu âm sau 1.2 giây
            setTimeout(() => {
                if(mediaRecorder && mediaRecorder.state === "recording") {
                    mediaRecorder.stop();
                }
            }, 1200);

        } catch (err) {
            alert("Lỗi kết nối Micro:\n" + err.message + "\n\nVui lòng đảm bảo bạn mở trang web này bằng Chrome/Safari và đã cho phép truy cập Micro.");
        }
    };

    // Hàm gửi File âm thanh + Cài đặt lên Server
    async function sendToServer(audioBlob) {
        const formData = new FormData();
        formData.append("file", audioBlob, "recording.wav");
        
        // Lấy giá trị từ các thanh trượt đưa vào Form để gửi lên
        formData.append("rms_thresh", sliderRms.value);
        formData.append("dt_thresh", sliderDt.value);
        formData.append("xn_thresh", sliderXn.value);

        try {
            const response = await fetch(API_URL, { 
                method: "POST", 
                body: formData 
            });
            
            const data = await response.json();

            if(data.error) {
                resultText.innerText = "❌ Lỗi Server";
                confText.innerText = data.error;
                resultText.style.color = "#e74c3c";
                return;
            }

            // Hiển thị kết quả
            const decision = data.decision;
            const confidence = data.confidence;
            const rms = data.raw_rms;

            confText.innerText = `Độ tin cậy: ${confidence}% (Raw: ${data.raw_label})`;
            rmsText.innerText = `RMS: ${rms.toFixed(5)}`;

            if(decision === "ĐỦ TUỔI") {
                resultText.innerText = "ĐỦ TUỔI 🍈";
                resultText.style.color = "#e67e22"; // Màu Cam
                // Rung điện thoại (Chỉ có tác dụng trên Android Chrome)
                if(navigator.vibrate) navigator.vibrate([200, 100, 200]); 
            } else if(decision === "XANH") {
                resultText.innerText = "XANH 🟢";
                resultText.style.color = "#27ae60"; // Màu Xanh lá
            } else if(decision === "NOISE") {
                resultText.innerText = "NỀN NHIỄU 🌊";
                resultText.style.color = "#3498db"; // Màu Xanh dương
            } else {
                resultText.innerText = "MỜI GÕ LẠI 🔄";
                resultText.style.color = "#95a5a6"; // Màu Xám
            }

        } catch (err) {
            resultText.innerText = "❌ Lỗi mạng";
            confText.innerText = "Không kết nối được đến AI Server";
            resultText.style.color = "#e74c3c";
        }
    }
</script>
</body>
</html>