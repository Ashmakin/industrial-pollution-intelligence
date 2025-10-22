#!/usr/bin/env python3
"""
测试CNEMC API响应
"""

import requests
import json

def test_cnemc_api():
    """测试CNEMC API"""
    base_url = "https://szzdjc.cnemc.cn:8070/GJZ/Ajax/Publish.ashx"
    
    # 测试参数
    params = {
        'AreaID': '110000',  # 北京
        'RiverID': '',
        'MNName': '',
        'PageIndex': -1,
        'PageSize': 100,
        'action': 'getRealDatas'
    }
    
    try:
        print(f"Testing CNEMC API with params: {params}")
        response = requests.get(base_url, params=params, timeout=30)
        print(f"Status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        # 尝试解析JSON
        try:
            data = response.json()
            print(f"JSON response type: {type(data)}")
            print(f"JSON response: {json.dumps(data, indent=2, ensure_ascii=False)}")
        except Exception as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {response.text[:500]}")
            
    except Exception as e:
        print(f"Request error: {e}")

if __name__ == "__main__":
    test_cnemc_api()
