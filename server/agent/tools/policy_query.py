"""保单信息查询工具"""
from __future__ import annotations

import logging
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

from configs.business_config import lis_url
from server.agent import model_container

# 单独运行的时候需要添加
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

_PROMPT_TEMPLATE = """
用户会提出一个关于保单信息查询的问题，你的目标是拆分出用户问题中的保单号，投保人名称 并按照我提供的工具回答。
例如 用户提出的问题是: 查询张三名下保单号为123456789的保单信息？
则 提取的保单号和投保人名称是: 123456789 张三
如果用户提出的问题是: 查询保单号为123456789的保单信息？
则 提取的保单号和投保人名称是: 123456789 None
如果用户提出的问题是: 查询投保人张三的保单信息？
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
产品名称：圆福终身重大疾病保险
保额：500000.0 
总保费：500000.0 
保单号：336.0
保险起期：2017-10-18 00:00:00
保险止期：2018-10-18 00:00:00
投保人名称：张三

被保人信息如下：
被保人姓名：猫猫
被保人电话： 18012451245

被保人姓名：猫猫
被保人电话： 18012451245

险种信息如下：
保单号：111111111112
险种名称： 轻症疾病保险金
险种保额：50000元

保单号：111111111112
险种名称： 轻症疾病保险金
险种保额：50000元

Answer: 以上是查询到的保单信息，请查收。

现在，这是我的问题：

问题: {question}
"""
PROMPT = PromptTemplate(
    input_variables=["question"],
    template=_PROMPT_TEMPLATE,
)


def query_policy(policy):
    apt_no, policy_no = split_query(policy)
    print("\n 入参数据如下： " + apt_no + "," + policy_no)

    base_url = lis_url + 'com.ifp.digitalbusiness/queryPolicyList'
    params = f"userId=" + apt_no
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    try:
        response = requests.post(base_url, headers=headers, data=params)
        data = response.json()
    except Exception as e:
        logging.exception("请求保单查询接口异常，以下返回默认值数据")
        data = {
            "plyDetail": {
                "applicant": {
                    "CAppCde": "",
                    "CAppNme": "纽带",
                    "CCertfCls": "0",
                    "CCertfCde": "611001198406065603",
                    "CClntMrk": "0",
                    "CMobile": "17621385669",
                    "CCounty": "310101",
                    "CProvince": "310000",
                    "CCity": "310100",
                    "CCountry": "",
                    "CSuffixAddr": "",
                    "CClntAddr": "咯哦微信我哦我",
                    "CZipCde": "123123",
                    "CEmail": "123@qq.com",
                    "CWorkDpt": "",
                    "CMrgCde": "",
                    "CRelCde": "32",
                    "CSex": "f",
                    "TBirthday": "1984-06-06",
                    "NAge": 33,
                    "COccType": "00 ",
                    "COccType2": "0001",
                    "COccType3": "0001001",
                    "CCusLvl": "1",
                    "CCertfDateSign": "1",
                    "CTaxCertfCde": "0",
                },
                "base": {
                    "CAppTyp": "A",
                    "CProdNo": "200001",
                    "CProdNme": "爱加保医疗保险(实惠款)",
                    "CDptCde": "99000000",
                    "CSlsCde": "1",
                    "CBsnsTyp": "1",
                    "CBsnsSubtyp": "1",
                    "CBrkrCde": "",
                    "CAgtAgrNo": "",
                    "NSubCoNo": "",
                    "NDiscRate": 0.0,
                    "CRenewMrk": "0",
                    "CAmtCur": "01",
                    "NAmt": 500000.0,
                    "NAmtRmbExch": 1.0,
                    "CPrmCur": "01",
                    "NCalcPrm": 1000.0,
                    "NPrm": 336.0,
                    "NPrmRmbExch": 1.0,
                    "TAppTm": 1508416379000,
                    "TInsrncBgnTm": "2017-10-18 00:00:00",
                    "TInsrncEndTm": "2018-10-18 00:00:00",
                    "CGrpMrk": "0",
                    "CInstMrk": "0",
                    "COprCde": "900001649",
                    "TOprTm": "2017-10-17 23:55:50",
                    "CFinTyp": "0",
                    "CSalegrpCde": "",
                    "CSlsId": "",
                    "CSlsNme": "",
                    "CSlsTel": "",
                    "CDataSrc": "02",
                    "NPayTime": 106,
                    "CPlanCde": "",
                    "CLegalBnfc": "1",
                    "CInsuYear": "1",
                    "CInsuYearFlag": "1",
                    "NPrmVar": 0.00,
                    "NNoTaxPrm": 316.98,
                    "NAddedTax": 19.02,
                    "NBasePrm": "200",
                    "CPayTimeFlag": "Y",
                    " CTmplFlag": "0",
                    "THsttPrd": "2017-10-28 00:00:00",
                    "CPkgSts": "1",
                    "CIsLineData": "1",
                    "NPaymentDay": 1080,
                    "NTimesPaid": "1",
                    "TNextPayDate": "2021-02-28",
                    "CIsRenewedFlag": "Y",
                    "NTotalPayPrm": 644.8,
                    "NPaidTime": 2,
                    "CNoticeNo": "123",
                    "NDssWtPrd": "5",
                    "CPkgNo": "61700206907",
                    "NHealthPrm": "",
                    "NPkgPrm": "",
                    "CFamilyFlag": "",
                    "CHealthUpgradeFlag": "",
                    "CDieDuty": "",
                    "CServiceStatus": ""
                },
                "bnfcList": [
                    {
                        "CBnfcCde": "",
                        "CBnfcNme": "猫猫",
                        "CInsuredCde": "",
                        "CRelCde": "31",
                        "CBenfOrd": "1",
                        "NBenfProp": 1,
                        "CCertfCde": "611001198406065603",
                        "CCertfCls": "0",
                        "CSex": "f",
                        "TBirthday": "1984-06-06",
                        "CClntMrk": "1",
                        "CMobile": "18012451245",
                        "CSuffixAddr": "",
                        "CCounty": "",
                        "CCity": "",
                        "CProvince": "",
                        "CCountry": "",
                        "CAddr": "纪录片可以在",
                        "CZipCde": "123123",
                        "CEmail": "",
                        "CBnfcType": "2",
                        "CCertfDateSign": "1"
                    }
                ],
                "cvrgList": [
                    {
                        "CCvrgNo": "200100",
                        "CCustCvrgNme": "一般医疗",
                        "NAmt": 500000.0,
                        "NBasePrm": 1000.0,
                        "NPrm": 309.0,
                        "CReMark": "",
                        "CCancelMrk": "1",
                        "TBgnTm": "2017-10-18 00:00:00",
                        "TEndTm": "2018-10-18 00:00:00",
                        "NNoTaxPrm": 291.51,
                        "NTaxRate": 0.06,
                        "NAddedTax": 17.49,
                        "NDayAmt": ""
                    },
                    {
                        "CCvrgNo": "200100",
                        "CCustCvrgNme": "一般医疗",
                        "NAmt": 500000.0,
                        "NRate": 309.0,
                        "NBasePrm": 1000.0,
                        "NPrm": 309.0,
                        "CReMark": "",
                        "NIndemLmt": 0.0,
                        "CCancelMrk": "1",
                        "TBgnTm": "2017-10-18 00:00:00",
                        "TEndTm": "2018-10-18 00:00:00",
                        "NNoTaxPrm": 291.51,
                        "NTaxRate": 0.06,
                        "NAddedTax": 17.49
                    },
                    {
                        "CCvrgNo": "200100",
                        "CCustCvrgNme": "一般医疗",
                        "NAmt": 500000.0,
                        "NBasePrm": 1000.0,
                        "NPrm": 309.0,
                        "CReMark": "",
                        "CCancelMrk": "1",
                        "TBgnTm": "2017-10-18 00:00:00",
                        "TEndTm": "2018-10-18 00:00:00",
                        "NPrmVar": 0.00,
                        "NNoTaxPrm": 291.51,
                        "NTaxRate": 0.06,
                        "NAddedTax": 17.49
                    }
                ],
                "insuredList": [
                    {
                        "CInsuredCde": "",
                        "CInsuredNme": "猫猫",
                        "CInsuredCls": "0",
                        "CRelInsuredCde": "",
                        "CCertfCde": "611001198406065603",
                        "CCertfCls": "0",
                        "CClntMrk": "0",
                        "CMobile": "18012451245",
                        "CCountry": "",
                        "CProvince": "610000",
                        "CCity": "610100",
                        "CCounty": "610101",
                        "CSuffixAddr": "",
                        "CClntAddr": "纪录片可以在",
                        "CZipCde": "123123",
                        "CEmail": "123@qq.cm",
                        "CWorkDpt": "",
                        "CWorkDptAddr": "",
                        "CWorkDptZip": "",
                        "CSex": "f",
                        "TBirthday": "1984-06-06",
                        "CMrgCde": "",
                        "CChldStsCde": "",
                        "NAge": 33.0,
                        "CEduLvlCde": "",
                        "CIsSoc": "1",
                        "COccType": "00 ",
                        "COccType2": "0001",
                        "COccType3": "0001001",
                        "CCusLvl": "1",
                        "CCertfDateSign": "1",
                        "NRevenue": 120000,
                        "CTwoHealInformFlag": "Y"
                    }
                ],
                "prodList": [
                    {
                        "CFinTyp": "12",
                        "CInsuYear": "106",
                        "CInsuYearFlag": "A",
                        "CProdNo": "CT1001",
                        "NPayTime": "20",
                        "CPayTimeFlag": "Y",
                        "CProdName": "瑞华个人医疗保险",
                        "NAmount": 2000000.0,
                        "NPrm": "282.0",
                        "TInsrncBgnTm": "2018-08-31 00:00:00",
                        "TInsrncEndTm": "2019-08-31 00:00:00"
                    }
                ],
                "riskList": [
                    {
                        "CAppNo": "91810251083720",
                        "CCrtCde": "Virtual",
                        "CCvrgNo": "301002",
                        "CEdrNo": "",
                        "CLatestMrk": "1",
                        "CPkId": "8a80813166ce57080166ce5729e90005",
                        "CPkgNo": "61800516427",
                        "CPlyNo": "111111111112",
                        "CProdNo": "300002",
                        "CReMark": "被保险人在等待期后初次罹患并经医院确诊为本合同定义的105种重大疾病，我们将按本合同基本保险金额给付重大疾病保险金，本合同终止。重大疾病确诊之后确诊的轻症疾病，我们不承担给付“轻症疾病保险金”的保险责任。同时确诊符合“重大疾病保险金”与“轻症疾病保险金”的给付条件，我们仅给付“重大疾病保险金”，本合同终止。重大疾病保险金的给付以一次为限。",
                        "CRiskCls": "1",
                        "CRiskNme": "重大疾病保险金",
                        "CRiskNo": "30000101",
                        "CRowId": "",
                        "CType": "01",
                        "CUnit": "元",
                        "CUpdCde": "Virtual",
                        "NAmt": "50000",
                        "NEdrPrjNo": 0,
                        "NSeqNo": 0,
                        "TCrtTm": "2018-11-01 16:15:11",
                        "TUpdTm": "2018-11-01 16:15:11"
                    },
                    {
                        "CAppNo": "91810251083720",
                        "CCrtCde": "Virtual",
                        "CCvrgNo": "301002",
                        "CEdrNo": "",
                        "CLatestMrk": "1",
                        "CPkId": "8a80813166ce57080166ce5729ea0006",
                        "CPkgNo": "61800516427",
                        "CPlyNo": "111111111112",
                        "CProdNo": "300002",
                        "CReMark": "首次轻症疾病保险金赔付30%的基本保额，该种轻症疾病的保险责任终止，合同继续有效；第2次轻症疾病保险金赔付35%的基本保额，该种轻症疾病的保险责任终止，合同继续有效；第3次轻症疾病保险金赔付40%的基本保额，该种轻症疾病的保险责任终止，此时轻症疾病保险金累积给付已达三次，轻症保险责任终止，重疾保险责任继续有效。同一轻症仅赔付1次，三次轻症保险金的赔付需要针对不同的轻症，且每次轻症需要在等待期后初次罹患并经医院确诊为本合同定义的55种轻症疾病。若被保险人因同一疾病原因或同一意外伤害事故导致其罹患本合同所定义的两种或两种以上的轻症疾病，我们仅按一种轻症疾病给付轻症疾病保险金。若我们已给付一次轻症疾病保险金，则本合同的现金价值自首次轻症疾病确诊之日起降低为零。",
                        "CRiskCls": "1",
                        "CRiskNme": "轻症疾病保险金",
                        "CRiskNo": "30000105",
                        "CRowId": "",
                        "CType": "01",
                        "CUnit": "元",
                        "CUpdCde": "Virtual",
                        "NAmt": "15000",
                        "NEdrPrjNo": 0,
                        "NSeqNo": 0,
                        "TCrtTm": "2018-11-01 16:15:11",
                        "TUpdTm": "2018-11-01 16:15:11"
                    },
                    {
                        "CAppNo": "91810251083720",
                        "CCrtCde": "Virtual",
                        "CCvrgNo": "301002",
                        "CEdrNo": "",
                        "CLatestMrk": "1",
                        "CPkId": "8a80813166ce57080166ce5729ea0007",
                        "CPkgNo": "61800516427",
                        "CPlyNo": "111111111112",
                        "CProdNo": "300002",
                        "CReMark": "被保险人在等待期后初次罹患并经医院确诊为本合同定义的轻症疾病，我们将豁免本合同的后续各期保险费，本项保险责任终止。被豁免的保险费视为已交纳，本合同继续有效，且本合同权益与正常交费的保险合同相同。",
                        "CRiskCls": "1",
                        "CRiskNme": "轻症疾病豁免保险费",
                        "CRiskNo": "30000106",
                        "CRowId": "",
                        "CType": "01",
                        "CUnit": "元",
                        "CUpdCde": "Virtual",
                        "NAmt": "豁免后期保险费",
                        "NEdrPrjNo": 0,
                        "NSeqNo": 0,
                        "TCrtTm": "2018-11-01 16:15:11",
                        "TUpdTm": "2018-11-01 16:15:11"
                    }
                ],
                "accountInfoList": [
                    {
                        "CType": "SQ",
                        "CBankCde": "104",
                        "CAcctNme": "验证",
                        "CAcctNo": "6216601152847617311"
                    },
                    {
                        "CType": "XQ",
                        "CBankCde": "BOC",
                        "CAcctNme": "验证",
                        "CAcctNo": "6216601152847617311"
                    }
                ]
            },
            "status": "success"
        }

    return format_policy_data(data)


def format_policy_data(data):
    status_ = data["status"]
    if status_ == 'success':
        formatted_data = f"\n 这是查询到的保单信息: \n"
        # 保单列表信息
        detail = data["plyDetail"]
        # 保单
        base = detail["base"]
        # 投保人
        applicant = detail["applicant"]

        # 产品名称
        formatted_data += '产品名称：' + base["CProdNme"] + '\n'
        # 保额
        formatted_data += '保额：' + str(base["NAmt"]) + '\n'
        # 总保费
        formatted_data += '总保费：' + str(base["NPrm"]) + '\n'
        # 保险起期
        formatted_data += '保险起期：' + base["TInsrncBgnTm"] + '\n'
        # 保险止期
        formatted_data += '保险止期：' + base["TInsrncEndTm"] + '\n'
        # 投保人名称
        formatted_data += '投保人名称：' + applicant["CAppNme"] + '\n'

        # 被保人信息
        formatted_data += '\n被保人信息如下：\n'
        insured_list = detail["insuredList"]
        for insured in insured_list:
            formatted_data += '被保人姓名：' + insured["CInsuredNme"] + '\n'
            formatted_data += '被保人电话：' + insured["CMobile"] + '\n'
            formatted_data += '\n'

        formatted_data += '\n险种信息如下：\n'
        # 险种信息
        risk_list = detail["riskList"]
        for risk in risk_list:
            formatted_data += '保单号：' + risk["CPlyNo"] + '\n'
            formatted_data += '险种名称：' + risk["CRiskNme"] + '\n'
            formatted_data += '险种保额：' + risk["NAmt"] + '\n'
            formatted_data += '\n'
    else:
        formatted_data = f"\n 没有查询到保单信息 \n"

    return formatted_data + "以上是查询到的保单信息，请查收\n"


def split_query(query):
    parts = query.split()
    adm = parts[0]
    if len(parts) == 1:
        return adm, adm
    location = parts[1] if parts[1] != 'None' else adm
    return location, adm


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
        return query_policy(expression)

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
