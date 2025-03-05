import os
import pandas as pd

# 定义文件夹路径
root_dir = '.'  # 根目录
control_dir = os.path.join(root_dir, 'Control')
dementia_dir = os.path.join(root_dir, 'Dementia')

# 任务名称和集中测试次数
tasks = ['cookie', 'fluency', 'recall', 'sentence']
test_sessions = range(6)  # 集中测试次数 0到5

# 用于存储最终的统计数据
data = {}

# 遍历Control和Dementia文件夹
for group_label, group_dir in [('Control', control_dir), ('Dementia', dementia_dir)]:
    # 遍历每个任务
    for task in tasks:
        task_dir = os.path.join(group_dir, task)
        # 遍历任务文件夹中的所有.cha文件
        for filename in os.listdir(task_dir):
            if filename.endswith('.cha'):
                # 提取被测试者ID和集中测试的次数
                participant_id, session_num = filename.split('-')
                session_num = session_num.split('.')[0]  # 获取集中测试次数

                # 如果这个ID还没有在数据字典中，初始化该ID的统计信息
                if participant_id not in data:
                    # 初始化新的一行数据
                    data[participant_id] = [participant_id, 0 if group_label == 'Control' else 1] + [0] * (len(tasks) * len(test_sessions))

                # 更新该被测试者对应的任务列信息
                task_index = tasks.index(task)
                session_index = test_sessions.index(int(session_num))
                column_index = task_index * len(test_sessions) + session_index

                # 标记该测试已参与
                data[participant_id][column_index + 2] = 1

# 将数据字典转换为DataFrame
columns = ['Participant ID', 'Group'] + [f'{task}_{session}' for task in tasks for session in test_sessions]
df = pd.DataFrame(list(data.values()), columns=columns)

# 输出为Excel文件
output_file = 'participant_test_data.xlsx'
df.to_excel(output_file, index=False)

print(f"数据已成功导出到 {output_file}")
