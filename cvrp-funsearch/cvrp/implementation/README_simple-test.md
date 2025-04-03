## 目前进度

完成了主要的函数流程，能够生成单个函数替换的流程
通过 cvrp_funsearch 的 main 流程可以进行测试

1. 首先加载 LLM 模型，然后读入数据（可以自定义地址）
2. 根据 prompt_config 中的 template_code,user_prompt 等加载 prompt_generator, evaluator
   - 这部分的自由度很高，是用来生成 prompt 的，可以自己 diy，加强对于 llm 的调用
   - TODO1: 这里的 config 后续可以抽象成一个类
   - TODO2: 后续可以考虑 evaluator 保留 evaluate 的功能，模版中抽掉 evaluate 函数的评估功能，直接返回 route（这样可以减少 token）
3. 调用 llm 生成回答
4. 将答案存入表(TODO3: 调用生成 csv 那个函数)
