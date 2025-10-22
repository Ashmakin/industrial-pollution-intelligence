#!/usr/bin/env python3
"""
自动数据采集调度器
每小时自动执行数据采集任务
"""

import time
import schedule
import logging
import subprocess
import os
import json
from datetime import datetime
from typing import Dict, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/auto_data_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoDataScheduler:
    """自动数据采集调度器"""
    
    def __init__(self):
        self.python_path = os.path.join(os.path.dirname(__file__), 'venv', 'bin', 'python3')
        self.collector_script = os.path.join(os.path.dirname(__file__), 'enhanced_cnemc_collector.py')
        self.db_url = os.getenv('DATABASE_URL', 'postgres://pollution_user:pollution_pass@localhost:5432/pollution_db')
        
        # 所有地区列表
        self.all_areas = [
            '北京', '天津', '河北', '山西', '内蒙古',
            '辽宁', '吉林', '黑龙江', '上海', '江苏',
            '浙江', '安徽', '福建', '江西', '山东',
            '河南', '湖北', '湖南', '广东', '广西',
            '海南', '重庆', '四川', '贵州', '云南',
            '西藏', '陕西', '甘肃', '青海', '宁夏', '新疆'
        ]
        
        # 分批采集，避免一次性请求过多
        self.batch_size = 5
        self.batches = [
            self.all_areas[i:i + self.batch_size] 
            for i in range(0, len(self.all_areas), self.batch_size)
        ]
    
    def run_data_collection(self, areas: List[str]) -> Dict:
        """执行数据采集"""
        try:
            logger.info(f"Starting data collection for areas: {areas}")
            
            # 构建命令
            areas_str = ','.join(areas)
            cmd = [
                self.python_path,
                self.collector_script,
                'collect',
                areas_str
            ]
            
            # 执行命令
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                # 解析结果
                try:
                    result_data = json.loads(result.stdout)
                    logger.info(f"Data collection completed: {result_data.get('total_inserted', 0)} inserted, {result_data.get('total_skipped', 0)} skipped")
                    return {
                        'success': True,
                        'data': result_data,
                        'timestamp': datetime.now().isoformat()
                    }
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse result: {result.stdout}")
                    return {
                        'success': False,
                        'error': 'Failed to parse result',
                        'timestamp': datetime.now().isoformat()
                    }
            else:
                logger.error(f"Data collection failed: {result.stderr}")
                return {
                    'success': False,
                    'error': result.stderr,
                    'timestamp': datetime.now().isoformat()
                }
                
        except subprocess.TimeoutExpired:
            logger.error("Data collection timed out")
            return {
                'success': False,
                'error': 'Timeout',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error running data collection: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def scheduled_collection_job(self):
        """定时采集任务"""
        logger.info("Starting scheduled data collection job")
        
        total_results = {
            'total_inserted': 0,
            'total_skipped': 0,
            'batch_results': [],
            'start_time': datetime.now().isoformat()
        }
        
        # 分批执行采集
        for i, batch in enumerate(self.batches):
            logger.info(f"Processing batch {i+1}/{len(self.batches)}: {batch}")
            
            result = self.run_data_collection(batch)
            total_results['batch_results'].append({
                'batch': i+1,
                'areas': batch,
                'result': result
            })
            
            if result['success']:
                data = result.get('data', {})
                total_results['total_inserted'] += data.get('total_inserted', 0)
                total_results['total_skipped'] += data.get('total_skipped', 0)
            
            # 批次间延迟，避免请求过于频繁
            if i < len(self.batches) - 1:
                logger.info("Waiting 30 seconds before next batch...")
                time.sleep(30)
        
        total_results['end_time'] = datetime.now().isoformat()
        total_results['duration'] = (
            datetime.fromisoformat(total_results['end_time']) - 
            datetime.fromisoformat(total_results['start_time'])
        ).total_seconds()
        
        logger.info(f"Scheduled collection completed: {total_results['total_inserted']} inserted, {total_results['total_skipped']} skipped")
        
        # 保存结果到文件
        result_file = f"/tmp/data_collection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(total_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Collection results saved to: {result_file}")
    
    def start_scheduler(self):
        """启动调度器"""
        logger.info("Starting auto data scheduler")
        
        # 设置定时任务 - 每小时执行一次
        schedule.every().hour.do(self.scheduled_collection_job)
        
        # 立即执行一次
        logger.info("Running initial data collection...")
        self.scheduled_collection_job()
        
        # 持续运行
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    
    def run_once(self):
        """执行一次数据采集"""
        logger.info("Running one-time data collection")
        self.scheduled_collection_job()

def main():
    """主函数"""
    import sys
    
    scheduler = AutoDataScheduler()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'once':
        # 执行一次采集
        scheduler.run_once()
    else:
        # 启动定时调度器
        scheduler.start_scheduler()

if __name__ == "__main__":
    main()
