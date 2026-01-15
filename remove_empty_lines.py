import sys
import os

def remove_empty_lines_basic(input_file, output_file=None):
    """
    去除文本文件中的空行（完全空的行）
    
    参数:
        input_file: 输入文件路径
        output_file: 输出文件路径，如果为None则覆盖原文件
    """
    try:
        # 读取文件内容
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 去除完全空白的行（包括只有换行符的行）
        cleaned_lines = [line for line in lines if line.strip() != '']
        
        # 确定输出文件路径
        if output_file is None:
            output_file = input_file
        
        # 写入清理后的内容
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        
        print(f"处理完成！原始行数: {len(lines)}, 处理后行数: {len(cleaned_lines)}")
        print(f"已保存到: {output_file}")
        
    except FileNotFoundError:
        print(f"错误: 文件 '{input_file}' 未找到")
    except Exception as e:
        print(f"处理文件时发生错误: {e}")

if __name__ == "__main__":
    # 使用示例
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        remove_empty_lines_basic(input_file, output_file)
    else:
        print("使用方法: python remove_empty_lines.py <输入文件> [输出文件]")
        print("如果只提供输入文件，将直接覆盖原文件")