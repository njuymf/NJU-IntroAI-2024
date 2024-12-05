import subprocess
import time


def run_and_log():
    log_file_path = 'game_results_LSTM.log'  # 日志文件的路径

    while True:
        # 执行原程序并捕获输出
        process = subprocess.Popen(
            ['python', 'test_LSTM.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # 等待程序完成并获取输出
        stdout, stderr = process.communicate()

        # 检查是否有错误输出
        if stderr:
            print(f"Error: {stderr}")

            # 查找最后一行输出
        last_line = ''
        for line in stdout.splitlines():
            print(f"Captured output: {line}")  # 添加调试信息，查看捕获的每一行
            if "Info:" in line:  # 查找包含"Info:"的行
                last_line = line

                # 检查是否有有效的最后一行
        if last_line:
            # 将最后一行写入日志文件
            with open(log_file_path, 'a') as log_file:
                log_file.write(last_line + '\n')
            print(f"Logged: {last_line}")
        else:
            print("No relevant output found, nothing logged.")

            # 等待一段时间后再执行
        time.sleep(0.1)


if __name__ == '__main__':
    run_and_log()
