import redis

# 连接到 Redis
client = redis.Redis(host="localhost", port=6379, decode_responses=True)

# 测试读写
client.set("test_key", "Hello, Redis!")
value = client.get("test_key")
print(f"Redis value: {value}")