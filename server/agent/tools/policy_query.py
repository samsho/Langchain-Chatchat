"""保单信息查询工具"""
from __future__ import annotations

import os
import re
import sys
import warnings
from typing import Dict
from typing import List, Any, Optional

import requests
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import Extra, root_validator
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from pydantic import BaseModel, Field

from server.agent import model_container

# 单独运行的时候需要添加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_PROMPT_TEMPLATE = """
用户会提出一个关于保单信息查询的问题，你的目标是拆分出用户问题中的保单号，投保人名称 并按照我提供的工具回答。
例如 用户提出的问题是: 请帮忙查询张三名下保单号为123456789的保单信息？
则 提取的保单号和投保人名称是: 123456789 张三
如果用户提出的问题是: 请帮忙查询保单号为123456789的保单信息？
则 提取的保单号和投保人名称是: 123456789 None
如果用户提出的问题是: 请帮忙查询投保人张三的保单信息？
则 提取的保单号和投保人名称是: None 张三
请注意以下内容:
1. 如果你没有找到投保人名称的内容,则一定要使用 None 替代，否则程序无法运行
2. 如果用户没有指定保单号，则一定要使用 None 替代，否则程序无法运行

问题: ${{用户的问题}}

你的回答格式应该按照下面的内容，请注意，格式内的```text 等标记都必须输出，这是我用来提取答案的标记。
```text

${{拆分的保单号和投保人名称，中间用空格隔开}}
```
... policy_query(保单号 投保人名称)...
```output

${{提取后的答案}}
```
答案: ${{答案}}



这是一个例子：
问题: 请帮忙查询张三名下保单号为123456789的保单信息？


```text
123456789 张三
```
...policy_query(123456789 张三)...

```output
保单号: 123456789
投保人: 张三
险种名称: 人寿保险
投保时间: 2023-09-11

Answer: 以上是查询到的保单信息，请查收。

现在，这是我的问题：

问题: {question}
"""
PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)


def get_policy_info(policyno, aptno):
    base_url = 'https://geoapi.qweather.com/v2/city/lookup?'
    params = {'policyno': policyno, 'aptno': aptno}
    response = requests.get(base_url, params=params)
    data = response.json()
    data = [{
        "policyno": "123456789",
        "aptno": "张三",
        "riskname": "人寿保险",
        "time": "2023-09-11"
    },
        {
            "policyno": "123456789",
            "aptno": "李四",
            "riskname": "人寿保险",
            "time": "2023-09-11"
        }]
    return format_policy_data(data)


def format_policy_data(data):
    formatted_data = f"\n 这是查询到的保单信息: \n"
    for forecast in data:
        formatted_data += '保单号: ' + forecast['policyno'] + '\n'
        formatted_data += '投保人: ' + forecast['aptno'] + '\n'
        formatted_data += '险种名称: ' + forecast['riskname'] + '\n'
        formatted_data += '投保时间: ' + forecast['time'] + '\n'
        formatted_data += '\n'
    return formatted_data


def split_query(query):
    parts = query.split()
    adm = parts[0]
    if len(parts) == 1:
        return adm, adm
    location = parts[1] if parts[1] != 'None' else adm
    return location, adm


def query_policy(policy):
    aptno, policyno = split_query(policy)
    print("\n 入参数据如下： " + aptno + "," + policy)
    try:
        policy_info = get_policy_info(policyno=policyno, aptno=aptno)
        return policy_info + "以上是查询到的保单信息，请查收\n"
    except KeyError:
        return "输入的保单信息不存在，无法提供信息查询"


class LLMPolicyChain(Chain):
    llm_chain: LLMChain
    llm: Optional[BaseLanguageModel] = None
    """[Deprecated] LLM wrapper to use."""
    prompt: BasePromptTemplate = PROMPT
    """[Deprecated] Prompt to use to translate to python if necessary."""
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @root_validator(pre=True)
    def raise_deprecation(cls, values: Dict) -> Dict:
        if "llm" in values:
            warnings.warn(
                "Directly instantiating an LLMWeatherChain with an llm is deprecated. "
                "Please instantiate with llm_chain argument or using the from_llm "
                "class method."
            )
            if "llm_chain" not in values and values["llm"] is not None:
                prompt = values.get("prompt", PROMPT)
                values["llm_chain"] = LLMChain(llm=values["llm"], prompt=prompt)
        return values

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _evaluate_expression(self, expression: str) -> str:
        try:
            output = query_policy(expression)
        except Exception as e:
            output = "输入的信息有误，请再次尝试"
        return output

    def _process_llm_result(
            self, llm_output: str, run_manager: CallbackManagerForChainRun
    ) -> Dict[str, str]:

        run_manager.on_text(llm_output, color="green", verbose=self.verbose)

        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)
        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            return {self.output_key: f"输入的格式不对: {llm_output},应该输入 (保单号 保单名称)的组合"}
        return {self.output_key: answer}

    async def _aprocess_llm_result(
            self,
            llm_output: str,
            run_manager: AsyncCallbackManagerForChainRun,
    ) -> Dict[str, str]:
        await run_manager.on_text(llm_output, color="green", verbose=self.verbose)
        llm_output = llm_output.strip()
        text_match = re.search(r"^```text(.*?)```", llm_output, re.DOTALL)

        if text_match:
            expression = text_match.group(1)
            output = self._evaluate_expression(expression)
            await run_manager.on_text("\nAnswer: ", verbose=self.verbose)
            await run_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif llm_output.startswith("Answer:"):
            answer = llm_output
        elif "Answer:" in llm_output:
            answer = "Answer: " + llm_output.split("Answer:")[-1]
        else:
            raise ValueError(f"unknown format from LLM: {llm_output}")
        return {self.output_key: answer}

    def _call(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(inputs[self.input_key])
        llm_output = self.llm_chain.predict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return self._process_llm_result(llm_output, _run_manager)

    async def _acall(
            self,
            inputs: Dict[str, str],
            run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(inputs[self.input_key])
        llm_output = await self.llm_chain.apredict(
            question=inputs[self.input_key],
            stop=["```output"],
            callbacks=_run_manager.get_child(),
        )
        return await self._aprocess_llm_result(llm_output, _run_manager)

    @property
    def _chain_type(self) -> str:
        return "llm_policy_chain"

    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            prompt: BasePromptTemplate = PROMPT,
            **kwargs: Any,
    ) -> LLMPolicyChain:
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)


def policy_query(policy: str):
    model = model_container.MODEL
    llm_policy = LLMPolicyChain.from_llm(model, verbose=True, prompt=PROMPT)
    ans = llm_policy.run(policy)
    return ans


class PolicySchema(BaseModel):
    policy: str = Field(description="应该是一个保单的信息，用空格隔开，保单号 投保人名称。例如：1908765 张三，如果没有投保人的信息，可以只输入 1908765")
