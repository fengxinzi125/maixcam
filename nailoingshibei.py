from maix import camera, display, image, nn, app, comm
import socket, struct, os

# ================= 1. 配置区 =================
SERVER_IP = '192.168.43.99' # 你的电脑 IP
SERVER_PORT = 1347          # VOFA+ 端口

# 跟随你的设定
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 320
CENTER_X = int(IMAGE_WIDTH / 2)  # 256
CENTER_Y = int(IMAGE_HEIGHT / 2) # 160

# ================= 2. AI 与 串口初始化 =================
APP_CMD_DETECT_RES = 0xAF

def encode_error_data(err_x, err_y):
    tail = 0xFF
    body = struct.pack("<BBhhB", 1, 0, err_x, err_y, tail)
    return body

# 加载模型
model_path = "model_276210.mud"
if not os.path.exists(model_path):
    model_path = "/root/models/test/model_276210.mud"
detector = nn.YOLOv5(model=model_path)

# 初始化串口协议
p = comm.CommProtocol(buff_size=1024)

# ================= 3. 网络连接 =================
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.settimeout(3.0) 
tcp_connected = False

try:
    print("正在连接 VOFA+...")
    client_socket.connect((SERVER_IP, SERVER_PORT))
    # 连上后把超时改短，防止发包卡顿拖慢了 AI 的速度
    client_socket.settimeout(0.05) 
    tcp_connected = True
    print("✅ 连接成功！")
except Exception as e:
    print(f"❌ 连接失败: {e}")
    print("⚠️ 程序将跳过图传，仅运行 AI 和单片机控制！")

# ================= 4. 摄像头与显示 =================
cam = camera.Camera(IMAGE_WIDTH, IMAGE_HEIGHT)
disp = display.Display()

# 截图原版的帧头格式
frame_head = [0, 0, IMAGE_WIDTH, IMAGE_HEIGHT, 13, 0x7F800000, 0x7F800000]

# ================= 5. 主循环 =================
while not app.need_exit():
    img = cam.read()
    
    # --- [新增] AI 推理与通信模块 ---
    objs = detector.detect(img, conf_th=0.79, iou_th=0.45)
    if len(objs) > 0:
        target = objs[0]     
        obj_cx = int(target.x + target.w / 2)
        obj_cy = int(target.y + target.h / 2)

        err_x = obj_cx - CENTER_X
        err_y = obj_cy - CENTER_Y

        # 1. 串口发送坐标给 STM32
        data_bytes = encode_error_data(err_x, err_y)
        p.report(APP_CMD_DETECT_RES, data_bytes)

        # 2. 画战术框 (必须在转 bytes 之前画，这样 VOFA 也能看到框)
        img.draw_cross(obj_cx, obj_cy, color=image.COLOR_GREEN, size=30)
        img.draw_rect(target.x, target.y, target.w, target.h, color=image.COLOR_RED, thickness=2)
        
    img.draw_cross(CENTER_X, CENTER_Y, color=image.COLOR_BLUE, size=20)
    
    disp.show(img)

    # --- [保留] 原汁原味的网络发送模块 ---
    if tcp_connected:
        try:
            # 1. 获取图片裸流
            image_bytes = img.to_bytes()
            
            # 2. 更新帧头里的数据长度
            frame_head[1] = len(image_bytes)

            # 3. 打包发送帧头
            buffer = b''
            for item in frame_head:
                buffer += struct.pack('<i', item)
            client_socket.sendall(buffer)

            # 4. 分块发送图片数据
            block_size = 2048
            block = int(len(image_bytes) / block_size)
            
            for i in range(block):
                client_socket.sendall(image_bytes[i * block_size : (i + 1) * block_size])
                
            # 发送尾巴数据
            client_socket.sendall(image_bytes[block * block_size : ])
            
        except Exception as e:
            # 遇到网络波动直接忽略本帧，继续跑循环，保护 AI 和单片机不掉线
            pass

print("程序退出")
if client_socket:
    client_socket.close()