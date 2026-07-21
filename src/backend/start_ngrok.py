from pyngrok import ngrok
import sys

# 你的 authtoken（已暴露没关系，这是免费版）
NGROK_AUTHTOKEN = "3GlIzj27u0LCuao4mQunTiRiINT_2mB9aRwVu4PbwhmoEMBHX"

def main():
    print("🔧 正在配置 ngrok...")
    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    
    print("🚀 正在启动隧道，转发到本地 8000 端口...")
    public_url = ngrok.connect(8000, "http")
    
    print("\n" + "=" * 60)
    print("✅ ngrok 隧道已启动！")
    print("=" * 60)
    print(f"\n📱 手机访问地址：")
    print(f"   {public_url}/mobile")
    print(f"\n💻 电脑监控地址：")
    print(f"   {public_url}/")
    print(f"\n⚠️  注意：免费版地址每次重启都会变化")
    print("=" * 60)
    print("\n按 Ctrl+C 停止隧道...")
    
    try:
        # 保持运行
        while True:
            input()
    except KeyboardInterrupt:
        print("\n⏹️  正在关闭隧道...")
        ngrok.kill()
        print("✅ 已关闭")

if __name__ == "__main__":
    main()