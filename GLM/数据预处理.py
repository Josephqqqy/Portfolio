import json

def process_jsonl_to_conversations(input_file, output_file):
    """
    将 JSONL 文件中的每个样本转换为对话格式
    
    Args:
        input_file: 输入的 JSONL 文件路径
        output_file: 输出的 JSONL 文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            try:
                # 解析每一行的 JSON 数据
                data = json.loads(line.strip())
                
                # 提取 input 和 output 字段
                user_content = data.get('input', '')
                assistant_content = data.get('output', '')
                
                # 构建目标格式
                result = {
                    "conversations": [
                        {
                            "role": "user",
                            "content": user_content
                        },
                        {
                            "role": "assistant",
                            "content": assistant_content
                        }
                    ]
                }
                
                # 写入输出文件
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行 JSON 解析错误: {e}")
                continue
            except Exception as e:
                print(f"第 {line_num} 行处理错误: {e}")
                continue
    
    print(f"处理完成！已处理 {line_num} 行数据")

if __name__ == "__main__":
    input_file = "DISC-Law-SFT-Triplet-released.jsonl"
    output_file = "DISC-Law-SFT-Triplet-conversations.jsonl"
    
    process_jsonl_to_conversations(input_file, output_file)

