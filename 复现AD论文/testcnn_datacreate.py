from tqdm import tqdm
import random
import os

# 显示安装提示
try:
    from tqdm import tqdm
except ImportError:
    print("请先安装进度条库：pip install tqdm")
    exit()

# 初始化参数
EMOTION_MAP = {
    "positive": 0,
    "negative": 1,
    "neutral": 2
}
OUTPUT_DIR = "dataset"
SENTENCE_PER_FILE = 5000  # 每个文件5000条，防止内存不足

# 扩展词汇表（关键部分）
vocab = {
    "positive_verbs": ["love", "enjoy", "admire", "celebrate", "appreciate"],
    "negative_verbs": ["hate", "dislike", "loathe", "can't stand", "avoid"],
    "neutral_verbs": ["have", "need", "know", "contain", "occur"],

    "positive_adjectives": ["amazing", "wonderful", "fantastic", "beautiful", "exciting"],
    "negative_adjectives": ["terrible", "awful", "horrible", "sad", "frustrating"],
    "neutral_adjectives": ["large", "old", "different", "important", "common"],

    # 新增情感强化词
    "intensifiers": {
        "positive": ["extremely", "absolutely", "truly", "so", "really"],
        "negative": ["completely", "totally", "extremely", "very", "quite"]
    }
}

# 情感模板库（按情感类型分类）
templates = {
    "positive": [
        "I {intensifier} {verb} {object}!",
        "What a {adj} {noun}!",
        "Everyone loves {place}'s {food}!",
        "{person} is {adj} at {activity}.",
        "I couldn't be happier about {event}"
    ],

    "negative": [
        "I {intensifier} {verb} {situation}.",
        "This {object} is {adj} to use.",
        "Why does {person} always {verb}?",
        "It's {adj} that {event} happened.",
        "I {verb} {experience} every time."
    ],

    "neutral": [
        "{subject} {verb} {detail}.",
        "In {location}, {fact} occurs {frequency}.",
        "The {measure} of {substance} is {value}.",
        "{historical_event} happened in {year}.",
        "A {animal} typically {behavior} during {season}."
    ]
}


# 生成带标签数据的函数
def generate_labeled_data(num_samples=10000):
    # 创建数据存储结构
    data = {"sentences": [], "labels": []}
    seen_sentences = set()  # 防止重复

    # 显示主进度条
    pbar = tqdm(total=num_samples, desc="Generating labeled sentences", unit="sentence")

    emotion_distribution = {
        "positive": int(num_samples * 0.6),
        "negative": int(num_samples * 0.3),
        "neutral": int(num_samples * 0.1)
    }

    for emotion in ["positive", "negative", "neutral"]:
        remaining = emotion_distribution[emotion]
        templates_list = templates[emotion]
        adj_list = vocab[f"{emotion}_adjectives"]
        verb_list = vocab[f"{emotion}_verbs"]
        intensifier_list = vocab["intensifiers"].get(emotion, [])

        # 组合所有可能的参数组合
        while remaining > 0 and len(data["sentences"]) < num_samples:
            # 随机选择模板
            template = random.choice(templates_list)

            # 解析模板占位符
            parts = template.split()
            new_parts = []
            for part in parts:
                if part.startswith("{"):
                    key = part[1:-1].lower().strip()
                    if key == "intensifier":
                        if intensifier_list:
                            new_parts.append(random.choice(intensifier_list))
                        else:
                            new_parts.append("")
                    elif key == "verb":
                        new_parts.append(random.choice(verb_list))
                    elif key == "adj":
                        new_parts.append(random.choice(adj_list))
                    elif key == "object":
                        # 对象词库需要扩展
                        object_options = [
                            "coffee", "movie", "book", "phone", "car",
                            "restaurant", "hotel", "flower", "music", "sport"
                        ]
                        new_parts.append(random.choice(object_options))
                    elif key == "situation":
                        situation_options = [
                            "waiting in line", "long commute", "broken device",
                            "meeting deadlines", "heavy workload"
                        ]
                        new_parts.append(random.choice(situation_options))
                    elif key == "place":
                        place_options = [
                            "Paris", "Tokyo", "New York", "Beijing", "London"
                        ]
                        new_parts.append(random.choice(place_options))
                    elif key == "food":
                        food_options = [
                            "sushi", "pizza", "tacos", "curry", "baklava"
                        ]
                        new_parts.append(random.choice(food_options))
                    elif key == "person":
                        person_options = [
                            "my boss", "my friend", "my neighbor", "my teacher",
                            "my colleague"
                        ]
                        new_parts.append(random.choice(person_options))
                    elif key == "subject":
                        subject_options = [
                            "Physics", "Chemistry", "Biology", "Mathematics",
                            "Economics"
                        ]
                        new_parts.append(random.choice(subject_options))
                    elif key == "event":
                        event_options = [
                            "birthday", "wedding", "concert", "vacation",
                            "promotion"
                        ]
                        new_parts.append(random.choice(event_options))
                    elif key == "location":
                        location_options = [
                            "library", "park", "museum", "gym", "cafe"
                        ]
                        new_parts.append(random.choice(location_options))
                    elif key == "food":
                        food_options = [
                            "steak", "sushi", "pasta", "salad", "dessert"
                        ]
                        new_parts.append(random.choice(food_options))
                    elif key == "animal":
                        animal_options = ["panda", "elephant", "giraffe", "koala", "penguin"]
                        new_parts.append(random.choice(animal_options))
                    elif key == "behavior":
                        behavior_options = ["hibernate", "migrate", "forage", "reproduce", "communicate"]
                        new_parts.append(random.choice(behavior_options))
                    elif key == "season":
                        season_options = ["spring", "summer", "autumn", "winter"]
                        new_parts.append(random.choice(season_options))
                    elif key == "year":
                        year_options = ["2020", "2021", "2022", "2023", "2024"]
                        new_parts.append(random.choice(year_options))
                    elif key == "frequency":
                        frequency_options = ["daily", "weekly", "monthly", "rarely", "often"]
                        new_parts.append(random.choice(frequency_options))
                    elif key == "measure":
                        measure_options = ["temperature", "speed", "weight", "height", "volume"]
                        new_parts.append(random.choice(measure_options))
                    elif key == "substance":
                        substance_options = ["water", "oxygen", "carbon", "salt", "sugar"]
                        new_parts.append(random.choice(substance_options))
                    elif key == "detail":
                        detail_options = [
                            "interesting facts", "complex theories", "beautiful patterns",
                            "specific methods", "accurate measurements"
                        ]
                        new_parts.append(random.choice(detail_options))
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)

            # 组合成完整句子
            sentence = ' '.join(new_parts).strip() + "."

            # 检查重复并更新进度
            if sentence not in seen_sentences:
                seen_sentences.add(sentence)
                data["sentences"].append(sentence)
                data["labels"].append(EMOTION_MAP[emotion])
                remaining -= 1
                pbar.update(1)  # 正确更新进度条

        pbar.close()  # 关闭进度条
        return data


# 生成并保存数据
def save_dataset(data):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 分块保存防止内存问题
    for i in range(0, len(data["sentences"]), SENTENCE_PER_FILE):
        chunk_sentences = data["sentences"][i:i + SENTENCE_PER_FILE]
        chunk_labels = data["labels"][i:i + SENTENCE_PER_FILE]

        # 保存句子文件
        with open(os.path.join(OUTPUT_DIR, f"sentence_{i // SENTENCE_PER_FILE}+1.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(chunk_sentences) + "\n")

        # 保存标签文件
        with open(os.path.join(OUTPUT_DIR, f"label_{i // SENTENCE_PER_FILE}+1.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(map(str, chunk_labels)) + "\n")


# 执行生成和保存
if __name__ == "__main__":
    print("开始生成带标签数据集...")
    dataset = generate_labeled_data(10000)
    save_dataset(dataset)
    print(f"\n🎉 数据集生成完成！共生成 {len(dataset['sentences'])} 条带标签句子")
    print(f"文件保存路径：{os.path.abspath(OUTPUT_DIR)}")