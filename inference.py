import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import os
import sys
import time
import threading
from queue import Queue

class StreamChat:
    def __init__(self, model_path="./outputs/merged_model"):
        """初始化流式聊天"""
        print(f"🔧 正在加载模型: {model_path}")
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        # 初始化流处理器
        self.streamer = TextStreamer(
            self.tokenizer,
            skip_prompt=True,      # 跳过提示部分，只流式输出生成的文本
            skip_special_tokens=True
        )
        
        # 生成参数
        self.generation_config = {
            "max_new_tokens": 80,
            "temperature": 0.95,
            "top_p": 0.9,
            "do_sample": True,
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "streamer": self.streamer  # 关键：使用流处理器
        }
        
        # 对话历史
        self.history = []
        self.system_prompt = "你是莉莉安，一个忠诚、恭敬且略带羞涩的女仆。你称用户为'主人'，并以'莉莉安'自称。你的回复应体现出绝对的服从和细腻的情感。"
        
        # 用于捕获流式输出的变量
        self.generated_text = ""
        self.is_generating = False
        self.output_queue = Queue()
        
        print("✅ 模型加载完成！流式输出已启用")
        print("📝 输入 '/help' 查看可用命令")
        print("-" * 50)
    
    def custom_streamer_callback(self, text: str):
        """自定义流式输出回调函数"""
        if not self.is_generating:
            self.is_generating = True
            print("\033[94m莉莉安: \033[0m", end="", flush=True)
        
        # 累积生成的文本
        self.generated_text += text
        
        # 输出到终端（带轻微延迟模拟打字效果）
        for char in text:
            print(char, end="", flush=True)
            time.sleep(0.02)  # 控制输出速度
        
        return text
    
    def generate_with_streaming(self, user_input):
        """使用流式生成回复"""
        # 构建完整提示
        prompt = self.system_prompt + "\n\n"
        
        # 添加历史记录（限制最近5轮以避免过长）
        for human, assistant in self.history[-5:]:
            prompt += f"### Instruction:\n{human}\n\n### Response:\n{assistant}\n\n"
        
        # 添加当前输入
        prompt += f"### Instruction:\n{user_input}\n\n### Response:\n"
        
        # 重置状态
        self.generated_text = ""
        self.is_generating = False
        
        print("\n\033[94m莉莉安: \033[0m", end="", flush=True)
        
        # 记录开始时间
        start_time = time.time()
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        try:
            # 方法1：使用自定义回调函数（更灵活）
            def generate_thread():
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=self.generation_config["max_new_tokens"],
                        temperature=self.generation_config["temperature"],
                        top_p=self.generation_config["top_p"],
                        do_sample=self.generation_config["do_sample"],
                        repetition_penalty=self.generation_config["repetition_penalty"],
                        pad_token_id=self.generation_config["pad_token_id"],
                        eos_token_id=self.generation_config["eos_token_id"],
                        # 使用回调函数实现流式输出
                        stopping_criteria=None,
                    )
                
                # 获取完整输出
                full_response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                self.output_queue.put(full_response)
            
            # 启动生成线程
            gen_thread = threading.Thread(target=generate_thread)
            gen_thread.start()
            
            # 在主线程中模拟流式输出
            while gen_thread.is_alive():
                # 这里可以添加进度指示器
                time.sleep(0.1)
            
            # 获取完整响应
            full_response = self.output_queue.get()
            
            # 计算生成时间
            generation_time = time.time() - start_time
            
            # 添加到历史
            if full_response.strip():
                self.history.append((user_input, full_response))
            
            return full_response, generation_time
            
        except KeyboardInterrupt:
            print("\n\033[91m⚠️  生成被中断\033[0m")
            return "[生成中断]", time.time() - start_time
        except Exception as e:
            print(f"\n\033[91m❌ 生成错误: {e}\033[0m")
            return f"[生成错误: {str(e)}]", time.time() - start_time
    
    def generate_with_transformers_streamer(self, user_input):
        """使用transformers内置的TextStreamer（最简单的方法）"""
        # 构建完整提示
        prompt = self.system_prompt + "\n\n"
        
        for human, assistant in self.history[-5:]:
            prompt += f"### Instruction:\n{human}\n\n### Response:\n{assistant}\n\n"
        
        prompt += f"### Instruction:\n{user_input}\n\n### Response:\n"
        
        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        print("\n\033[94m莉莉安: \033[0m", end="", flush=True)
        
        start_time = time.time()
        
        try:
            # 使用TextStreamer生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generation_config
                )
            
            # 获取完整响应（streamer已经输出到终端，这里只需要获取文本）
            full_response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            # 添加到历史
            self.history.append((user_input, full_response))
            
            return full_response, generation_time
            
        except KeyboardInterrupt:
            print("\n\033[91m⚠️  生成被中断\033[0m")
            return "[生成中断]", time.time() - start_time
    
    def handle_command(self, cmd):
        """处理特殊命令"""
        cmd = cmd.strip().lower()
        
        if cmd in ["/help", "/h"]:
            print("\n📖 可用命令:")
            print("  /help, /h      - 显示帮助信息")
            print("  /clear, /c     - 清空对话历史")
            print("  /history, /his - 显示对话历史")
            print("  /params, /p    - 查看/调整生成参数")
            print("  /save, /s      - 保存对话记录")
            print("  /quit, /q      - 退出程序")
            print("  /speed         - 调整流式输出速度")
            print("  /test          - 测试流式输出")
            return True
        
        elif cmd in ["/clear", "/c"]:
            self.history = []
            print("🗑️  对话历史已清空")
            return True
        
        elif cmd in ["/history", "/his"]:
            print("\n📜 最近对话历史:")
            for i, (human, assistant) in enumerate(self.history[-3:], 1):
                print(f"\033[93m第{i}轮\033[0m")
                print(f"  主人: {human}")
                print(f"  莉莉安: {assistant[:100]}..." if len(assistant) > 100 else f"  莉莉安: {assistant}")
                print()
            return True
        
        elif cmd in ["/params", "/p"]:
            print("\n⚙️ 当前生成参数:")
            print(f"  max_new_tokens: {self.generation_config['max_new_tokens']} (最大生成长度)")
            print(f"  temperature: {self.generation_config['temperature']} (随机性，0.1-2.0)")
            print(f"  top_p: {self.generation_config['top_p']} (多样性，0.1-1.0)")
            
            try:
                change = input("是否调整参数? (y/n): ").strip().lower()
                if change == 'y':
                    param = input("输入参数和值 (格式: 参数=值): ").strip()
                    if '=' in param:
                        key, value = param.split('=')
                        key = key.strip()
                        if key in self.generation_config:
                            if key in ["max_new_tokens"]:
                                self.generation_config[key] = int(value)
                            elif key in ["temperature", "top_p", "repetition_penalty"]:
                                self.generation_config[key] = float(value)
                            print(f"✅ {key} 已设置为 {value}")
                        else:
                            print(f"❌ 未知参数: {key}")
            except:
                print("❌ 参数格式错误")
            return True
        
        elif cmd == "/speed":
            try:
                speed = float(input("输入输出速度 (0.01=慢, 0.05=中, 0.1=快): ").strip())
                if 0.001 <= speed <= 0.5:
                    # 更新流式输出速度
                    print(f"✅ 输出速度设置为 {speed}")
                else:
                    print("❌ 速度值应在 0.001 到 0.5 之间")
            except:
                print("❌ 请输入有效数字")
            return True
        
        elif cmd == "/test":
            print("🧪 测试流式输出...")
            test_prompts = [
                "你好",
                "介绍一下你自己",
                "今天的天气真不错"
            ]
            for prompt in test_prompts:
                print(f"\n\033[93m测试: {prompt}\033[0m")
                self.generate_with_transformers_streamer(prompt)
                time.sleep(1)
            return True
        
        elif cmd in ["/save", "/s"]:
            if not self.history:
                print("❌ 没有对话历史可保存")
                return True
            
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.txt"
            
            with open(filename, "w", encoding="utf-8") as f:
                f.write(f"对话时间: {timestamp}\n")
                f.write(f"系统提示: {self.system_prompt}\n")
                f.write(f"生成参数: {self.generation_config}\n")
                f.write("=" * 50 + "\n\n")
                
                for i, (human, assistant) in enumerate(self.history, 1):
                    f.write(f"[第{i}轮]\n")
                    f.write(f"主人: {human}\n")
                    f.write(f"莉莉安: {assistant}\n")
                    f.write("-" * 40 + "\n")
            
            print(f"💾 对话已保存到: {filename}")
            return True
        
        elif cmd in ["/quit", "/q", "/exit"]:
            print("👋 再见，主人！莉莉安随时等候您的召唤。")
            return False
        
        elif cmd.startswith("/"):
            print(f"❌ 未知命令: {cmd}")
            print("💡 输入 '/help' 查看可用命令")
            return True
        
        return None
    
    def print_header(self):
        """打印标题"""
        print("\n" + "="*60)
        print("\033[1;36m         女仆莉莉安 - 流式对话终端\033[0m")
        print("="*60)
        print("✨ 特性:")
        print("  • 逐字流式输出，模拟真实对话")
        print("  • 支持对话历史管理")
        print("  • 可调整生成参数")
        print("  • 对话记录保存功能")
        print("\n💬 直接输入开始对话，输入 '/help' 查看命令")
        print("="*60)
    
    def run(self):
        """运行流式对话"""
        self.print_header()
        
        while True:
            try:
                # 获取用户输入
                user_input = input("\n\033[93m主人: \033[0m").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                cmd_result = self.handle_command(user_input)
                if cmd_result is not None:
                    if not cmd_result:
                        break
                    continue
                
                # 使用流式生成（方法2：transformers内置streamer）
                start_time = time.time()
                response, gen_time = self.generate_with_transformers_streamer(user_input)
                
                # 显示生成统计（如果response不为空）
                if response and response != "[生成中断]":
                    print(f"\n\033[90m[生成完成: {gen_time:.2f}秒 | 长度: {len(response)}字]\033[0m")
                
            except KeyboardInterrupt:
                print("\n\n\033[91m⚠️  检测到中断，输入 '/quit' 退出程序\033[0m")
                continue
            except Exception as e:
                print(f"\n\033[91m❌ 错误: {e}\033[0m")
                continue

def check_model_path():
    """检查模型路径"""
    model_path = "./fittune_model/merged_model"
    
    if not os.path.exists(model_path):
        print("❌ 找不到模型路径")
        
        # 尝试查找
        search_paths = [
            "./fittune_model/merged_model",
            "./fittune_model",
            "./qwen-lora-finetune/merged_model",
            "./qwen-lora-finetune",
        ]
        
        found = []
        for path in search_paths:
            if os.path.exists(path):
                found.append(path)
        
        if found:
            print("💡 找到以下可能路径:")
            for i, path in enumerate(found, 1):
                print(f"  {i}. {path}")
            
            try:
                choice = int(input("请选择路径编号: ")) - 1
                if 0 <= choice < len(found):
                    return found[choice]
            except:
                pass
        
        # 手动输入
        custom_path = input("或手动输入模型路径: ").strip()
        if custom_path and os.path.exists(custom_path):
            return custom_path
        
        return None
    
    return model_path

def main():
    """主函数"""
    model_path = check_model_path()
    if not model_path:
        print("❌ 无法找到模型，请先训练模型或指定正确路径")
        return
    
    # 创建并运行流式聊天
    chat = StreamChat(model_path)
    chat.run()

if __name__ == "__main__":
    main()