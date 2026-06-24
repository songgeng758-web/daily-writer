import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI

from daily_writer import SYSTEM_PROMPT, build_user_prompt


load_dotenv()


def get_env_value(name):
    """读取环境变量，兼容本地 .env 和 Vercel 环境变量。"""
    return os.getenv(name, "").strip()


def json_response(handler, status_code, data):
    """返回 JSON 数据给前端页面。"""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status_code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def save_report(report):
    """本地运行时保存到 output.txt，Vercel 上保存到临时目录。"""
    output_path = "/tmp/output.txt" if os.getenv("VERCEL") else "output.txt"
    with open(output_path, "w", encoding="utf-8") as file:
        file.write(report)
        file.write("\n")


def generate_report(today_text):
    """调用 DeepSeek 生成日报内容。"""
    api_key = get_env_value("DEEPSEEK_API_KEY")
    base_url = get_env_value("DEEPSEEK_BASE_URL")
    model = get_env_value("DEEPSEEK_MODEL")

    if not api_key or not base_url or not model:
        raise RuntimeError("服务端环境变量未配置完整，请检查 Vercel 的 Environment Variables。")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(today_text)},
        ],
        temperature=0.5,
    )

    if not response.choices:
        raise RuntimeError("模型没有返回有效内容。")

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError("模型返回内容为空。")

    report = content.strip()
    save_report(report)
    return report


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """访问首页时返回前端页面。"""
        if self.path not in ["/", "/index.html"]:
            json_response(self, 404, {"error": "页面不存在。"})
            return

        try:
            index_path = Path(__file__).resolve().parent.parent / "index.html"
            body = index_path.read_text(encoding="utf-8").encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as error:
            json_response(self, 500, {"error": f"首页加载失败：{error}"})

    def do_POST(self):
        if self.path != "/api/generate":
            json_response(self, 404, {"error": "接口不存在。"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length).decode("utf-8")
            data = json.loads(body) if body else {}
            today_text = str(data.get("today_text", "")).strip()

            if not today_text:
                json_response(self, 400, {"error": "请先输入今天大致做了什么。"})
                return

            report = generate_report(today_text)
            json_response(self, 200, {"report": report})
        except AuthenticationError:
            json_response(self, 500, {"error": "DeepSeek API Key 验证失败，请检查 Vercel 环境变量。"})
        except APIConnectionError:
            json_response(self, 500, {"error": "网络连接失败，请检查 DeepSeek 地址或稍后重试。"})
        except APIError as error:
            json_response(self, 500, {"error": f"模型接口返回失败：{error}"})
        except Exception as error:
            json_response(self, 500, {"error": f"生成日报失败：{error}"})
