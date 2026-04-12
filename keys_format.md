请新建一个keys.json文件，按如下格式配置：

``` json
{
  "hf_token": "YOUR_HF_TOKEN"
}
```

- `hf_token`: Hugging Face 访问令牌，用于下载模型和数据集

说明：
- 当前评测框架（C-Eval、IFEval、BFCL v4）均不需要外部 LLM API 进行评判
- 所有评测都是基于规则的确定性评测，使用本地 vLLM 服务提供推理