import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, APIConnectionError, APIError, AuthenticationError


OUTPUT_FILE = "output.txt"


SYSTEM_PROMPT = """你是一名企业 HCM/ERP 实施顾问日报助手。

用户在郑州浪潮信息技术公司做实施顾问相关工作，日常工作包括 HCM 系统学习、组织人事模块操作、人员信息维护、岗位任职、报表平台、测试环境验证、JSON 数据源配置、字段筛选、问题复现、数据验证、方案整理和导师沟通。

请根据用户输入的大白话，生成适合填写到公司日报系统的内容。

要求：
1. 严格按照 5 个栏目输出：
   今日工作：
   重要事情事项进展：
   我的观点和洞察：
   需要资源和协作：
   明日计划：
2. 每个栏目 1 到 3 句话即可。
3. 语言自然，像刚入职不久的实施顾问自己写的，不要太 AI，不要像宣传稿。
4. 语气认真、踏实、克制，可以稍微口语化，但不要太学生气。
5. 优先围绕用户真实输入整理，不要擅自编造重大成果、上线结果、客户结论或复杂方案。
6. 可以适当体现“测试环境验证、问题定位、配置调整、数据核对、向导师反馈、继续学习”等实施顾问工作特点。
7. 工作描述要具体一点，少写空话；如果只是学习或跟着导师处理，也可以如实表达。
8. “我的观点和洞察”要写成当天工作中的小体会，不要拔高到战略层面。
9. “明日计划”要承接今天内容，写可执行的小计划。
10. 如果用户没有提到资源协作，可以写“暂无额外资源需求，后续根据导师反馈继续调整。”这类自然表达。
11. 不要使用 Markdown 标题、序号、项目符号或加粗符号，只输出纯文本。
"""


def get_env_value(name):
    """读取环境变量，去掉首尾空格，避免误填空值。"""
    return os.getenv(name, "").strip()


def check_config():
    """检查 DeepSeek API 配置是否完整。"""
    api_key = get_env_value("DEEPSEEK_API_KEY")
    base_url = get_env_value("DEEPSEEK_BASE_URL")
    model = get_env_value("DEEPSEEK_MODEL")

    missing_items = []
    if not api_key:
        missing_items.append("DEEPSEEK_API_KEY")
    if not base_url:
        missing_items.append("DEEPSEEK_BASE_URL")
    if not model:
        missing_items.append("DEEPSEEK_MODEL")

    if missing_items:
        print("配置不完整，请先在 .env 文件中配置以下内容：")
        for item in missing_items:
            print(f"- {item}")
        print("\n可以参考 .env.example 文件填写。")
        sys.exit(1)

    return api_key, base_url, model


def build_user_prompt(today_text):
    """把用户的大白话包装成更明确的生成任务。"""
    return f"""今天大致做了什么：
{today_text}

请整理为公司日报系统可直接填写的内容。输出格式必须严格如下：

今日工作：
……

重要事情事项进展：
……

我的观点和洞察：
……

需要资源和协作：
……

明日计划：
……
"""


def generate_daily_report(client, model, today_text):
    """调用 DeepSeek 的 OpenAI 兼容接口生成日报。"""
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

    return content.strip()


def save_report(report):
    """把日报保存到本地文件。"""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:
        file.write(report)
        file.write("\n")


def main():
    load_dotenv()

    api_key, base_url, model = check_config()

    print("请输入今天大致做了什么，可以直接用大白话描述：")
    today_text = input("> ").strip()

    if not today_text:
        print("你还没有输入今天的工作内容，请重新运行后再试。")
        sys.exit(1)

    client = OpenAI(api_key=api_key, base_url=base_url)

    try:
        report = generate_daily_report(client, model, today_text)
        save_report(report)

        print("\n生成结果：\n")
        print(report)
        print(f"\n已保存到 {OUTPUT_FILE}")
    except AuthenticationError:
        print("DeepSeek API Key 验证失败，请检查 .env 中的 DEEPSEEK_API_KEY 是否正确。")
        sys.exit(1)
    except APIConnectionError:
        print("网络连接失败，请检查网络是否正常，或 DEEPSEEK_BASE_URL 是否填写正确。")
        sys.exit(1)
    except APIError as error:
        print(f"模型接口返回失败：{error}")
        sys.exit(1)
    except Exception as error:
        print(f"生成日报失败：{error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
