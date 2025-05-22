from agent import CodeGenAgent

if __name__ == "__main__":
    prompt = "写一个线性回归的代码"
    agent = CodeGenAgent()
    agent.run(prompt)