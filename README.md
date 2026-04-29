Advanced Driver Monitoring System (DMS)
Hệ thống giám sát trạng thái tài xế thời gian thực sử dụng trí tuệ nhân tạo để phát hiện các dấu hiệu mệt mỏi, buồn ngủ và mất tập trung. Đây là sản phẩm thuộc đồ án tốt nghiệp ngành Khoa học máy tính tại Đại học Công nghiệp Hà Nội (HaUI).

Tính năng nổi bật:
    Nhận diện buồn ngủ (Drowsiness): Sử dụng chỉ số EAR (Eye Aspect Ratio) kết hợp mạng MobileNetV2 để phát hiện trạng thái nhắm mắt hoặc ngủ gật vi mô (Micro-sleep).

    Phát hiện ngáp (Yawning): Dung hợp chéo giữa chỉ số MAR (Mouth Aspect Ratio) và mô hình học sâu để cảnh báo hành vi ngáp mệt mỏi.

    Cảnh báo mất tập trung (Distraction): Theo dõi hướng đầu (Head Ratio) và tính toán Attention Score để phát hiện việc tài xế cúi đầu hoặc nhìn xuống điện thoại.

    Nhận diện cảm xúc (Road Rage): Sử dụng mạng ResNet50 để nhận diện trạng thái tức giận, giúp ngăn chặn các hành vi lái xe bạo lực.

Tối ưu hóa thời gian thực:
    Multi-threading: Tách luồng Camera độc lập để triệt tiêu độ trễ I/O.

    Resolution Decoupling: Xử lý AI trên khung hình thấp nhưng hiển thị giao diện HUD chuẩn HD.

    Frame Skipping: Tối ưu hóa chu kỳ nội suy AI để duy trì FPS ổn định.

Kiến trúc hệ thống:
    Dự án được thiết kế theo cấu trúc mô-đun (Modular Design) giúp dễ dàng bảo trì và nâng cấp:
        main_system.py: Bộ điều khiển trung tâm (Controller) và giao diện người máy (HMI).

        core_ai.py: Quản lý việc nạp và nội suy các mô hình PyTorch (MobileNetV2, ResNet50).

        core_vision.py: Xử lý các phép toán hình học sinh trắc học và luồng Camera đa luồng.

        config.py: Quản lý toàn bộ hằng số, ngưỡng (thresholds) và cấu hình hệ thống.

Cài đặt & Sử dụng:
    Yêu cầu hệ thống:
        Python 3.8 hoặc cao hơn.

        Camera (Webcam) hoạt động ổn định.

        Phông chữ arial.ttf và arialbd.ttf trong hệ thống.

    Cài đặt thư viện:
Bash
pip install -r requirements.txt

    Tải trọng số mô hình (Pre-trained Weights):
        Do giới hạn dung lượng Git, vui lòng tải các file trọng số tại mục Releases và đặt vào thư mục weights/:
            best_eye_state_mobilenetv2.pt

            best_yawn_ultimate.pt

            best_emotion_model.pt

    Chạy hệ thống
Bash
python main_system.py

Phương pháp tiếp cận:
    Hệ thống áp dụng cơ chế Dynamic Calibration (Hiệu chỉnh cá nhân hóa) trong 1.5 giây đầu tiên khi khởi động để tự động học ngưỡng EAR và tư thế ngồi của từng tài xế, đảm bảo độ chính xác tối đa cho mọi đối tượng sử dụng.

Tác giả: Trần Xuân Đức - Sinh viên ngành Khoa học Máy tính - Đại học Công nghiệp Hà Nội.

Giảng viên hướng dẫn: TS. Nguyễn Mạnh Cường.