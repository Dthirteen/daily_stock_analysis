# -*- coding: utf-8 -*-
"""
===================================
板块及个股分析模块
===================================

职责：
1. 增强板块数据，包括资金量和权重计算
2. 获取板块领头个股数据
3. 根据板块表现选出5支值得关注的股票
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SectorWithCapital:
    """板块数据（包含资金量）"""
    name: str                      # 板块名称
    code: Optional[str] = None     # 板块代码
    change_pct: float = 0.0        # 涨跌幅(%)
    change: float = 0.0            # 涨跌点数
    volume: float = 0.0            # 成交量
    amount: float = 0.0            # 成交额（亿元）
    stock_count: int = 0           # 板块内股票数
    up_count: int = 0              # 上涨股票数
    leader_code: Optional[str] = None  # 龙头股代码
    leader_name: Optional[str] = None  # 龙头股名称
    leader_change_pct: float = 0.0     # 龙头股涨跌幅
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'code': self.code,
            'change_pct': self.change_pct,
            'change': self.change,
            'volume': self.volume,
            'amount': self.amount,
            'stock_count': self.stock_count,
            'up_count': self.up_count,
            'leader_code': self.leader_code,
            'leader_name': self.leader_name,
            'leader_change_pct': self.leader_change_pct,
        }


@dataclass
class StockRecommendation:
    """股票推荐项"""
    code: str
    name: str
    price: float = 0.0
    change_pct: float = 0.0
    sector: str = ""               # 所属板块
    reason: str = ""               # 推荐理由
    confidence: float = 0.0        # 信心度 0-100
    # ── 扩展字段（智能选股）──
    pick_source: str = ""          # 来源：limit_up(涨停池) / sector_leader(板块龙头) / hot_sector(热门板块)
    turnover_rate: float = 0.0     # 换手率
    amount: float = 0.0            # 成交额（亿）
    tech_score: float = 0.0        # 技术面评分 0-100
    is_first_limit: bool = False   # 是否一字板
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'price': self.price,
            'change_pct': self.change_pct,
            'sector': self.sector,
            'reason': self.reason,
            'confidence': self.confidence,
            'pick_source': self.pick_source,
            'turnover_rate': self.turnover_rate,
            'amount': self.amount,
            'tech_score': self.tech_score,
        }


class MarketSectorAnalyzer:
    """
    板块及个股分析器
    
    功能：
    1. 增强板块数据（添加资金量、龙头股等信息）
    2. 计算板块资金量权重
    3. 推荐值得关注的个股
    """
    
    def __init__(self, data_manager):
        """
        初始化分析器
        
        Args:
            data_manager: DataFetcherManager 实例
        """
        self.data_manager = data_manager
        
    def get_sector_rankings_with_capital(self, n: int = 5) -> Tuple[List[SectorWithCapital], List[SectorWithCapital]]:
        """
        获取板块涨跌榜（包含资金量和龙头股信息）
        
        Returns:
            Tuple: (领涨板块列表, 领跌板块列表)
        """
        logger.info("[板块分析] 获取板块涨跌榜...")
        
        try:
            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(n)
            
            # 转换为 SectorWithCapital 对象
            top_list = [self._enrich_sector(s) for s in top_sectors]
            bottom_list = [self._enrich_sector(s) for s in bottom_sectors]
            
            logger.info(f"[板块分析] 获取成功，领涨 {len(top_list)} 个，领跌 {len(bottom_list)} 个")
            return top_list, bottom_list
            
        except Exception as e:
            logger.error(f"[板块分析] 获取板块排行失败: {e}")
            return [], []
    
    def _enrich_sector(self, sector_dict: Dict[str, Any]) -> SectorWithCapital:
        """
        增强板块数据
        
        将原始板块字典转换为 SectorWithCapital，添加额外信息
        """
        sector = SectorWithCapital(
            name=sector_dict.get('name', ''),
            code=sector_dict.get('code'),
            change_pct=float(sector_dict.get('change_pct', 0.0)),
            change=float(sector_dict.get('change', 0.0)),
            volume=float(sector_dict.get('volume', 0.0)),
            amount=float(sector_dict.get('amount', 0.0)),  # 通常已转为亿元
            stock_count=int(sector_dict.get('stock_count', 0)),
            up_count=int(sector_dict.get('up_count', 0)),
        )
        
        # 尝试获取龙头股信息
        leader = self._get_sector_leader(sector.name)
        if leader:
            sector.leader_code = leader.get('code')
            sector.leader_name = leader.get('name')
            sector.leader_change_pct = float(leader.get('change_pct', 0.0))
        
        return sector
    
    def _get_sector_leader(self, sector_name: str) -> Optional[Dict[str, Any]]:
        """
        获取板块龙头股（涨幅最大的股票）

        通过 DataFetcherManager.get_sector_constituents 获取板块成分股，
        然后选取涨跌幅最大的作为龙头股。
        """
        try:
            # 调用数据源获取板块成分股
            df = self.data_manager.get_sector_constituents(sector_name)
            if df is None or df.empty:
                logger.debug(f"[板块分析] 板块 {sector_name} 成分股为空")
                return None

            # 确保 change_pct 列存在且为数值类型
            if 'change_pct' not in df.columns:
                logger.debug(f"[板块分析] 板块 {sector_name} 数据缺少 change_pct 列")
                return None

            df['change_pct'] = pd.to_numeric(df['change_pct'], errors='coerce')
            df = df.dropna(subset=['change_pct'])

            if df.empty:
                return None

            # 取涨幅最大的一只作为龙头股
            leader_idx = df['change_pct'].idxmax()
            leader = df.loc[leader_idx]

            result = {
                'code': str(leader.get('code', '')),
                'name': leader.get('name', ''),
                'change_pct': float(leader['change_pct']),
            }

            # 补充可选字段
            for extra_key in ('price', 'turnover_rate', 'amount', 'market_cap'):
                if extra_key in leader.columns or extra_key in leader.index:
                    result[extra_key] = leader.get(extra_key)

            logger.info(
                f"[板块分析] {sector_name} 龙头股: {result['name']}({result['code']}) "
                f"涨幅 {result['change_pct']:+.2f}%"
            )
            return result

        except Exception as e:
            logger.warning(f"[板块分析] 获取板块 {sector_name} 龙头股失败: {e}")
            return None
    
    def calculate_sector_capital_weights(
        self,
        top_sectors: List[SectorWithCapital],
        bottom_sectors: List[SectorWithCapital]
    ) -> Dict[str, float]:
        """
        计算板块资金量权重
        
        按成交额占比计算权重，用于评估板块热度
        
        Returns:
            {板块名: 权重} 字典，权重 0-100
        """
        all_sectors = top_sectors + bottom_sectors
        if not all_sectors:
            return {}
        
        total_amount = sum(s.amount for s in all_sectors if s.amount > 0)
        if total_amount <= 0:
            return {}
        
        weights = {}
        for sector in all_sectors:
            if sector.amount > 0:
                weight = (sector.amount / total_amount) * 100
                weights[sector.name] = round(weight, 2)
        
        return weights
    
    def get_recommended_stocks(
        self,
        top_sectors: List[SectorWithCapital],
        max_stocks: int = 5,
        min_confidence: float = 0.5
    ) -> List[StockRecommendation]:
        """
        推荐值得关注的个股
        
        策略：
        1. 优先选择涨幅最大、资金量最大的板块中的龙头股
        2. 选择涨幅最大的个股（可根据热门板块扩展搜索）
        3. 计算推荐信心度
        
        Args:
            top_sectors: 领涨板块列表
            max_stocks: 最多推荐股票数
            min_confidence: 最小信心度阈值
            
        Returns:
            StockRecommendation 对象列表
        """
        recommendations = []
        
        logger.info(f"[个股推荐] 开始分析，目标 {max_stocks} 支股票...")
        
        # 策略1：从领涨板块的龙头股中筛选
        for sector in top_sectors[:3]:  # 只看前3个领涨板块
            if len(recommendations) >= max_stocks:
                break
                
            if sector.leader_code and sector.leader_name:
                # 龙头股信心度较高
                confidence = min(100, 70 + sector.change_pct)
                if confidence >= min_confidence * 100:
                    stock = StockRecommendation(
                        code=sector.leader_code,
                        name=sector.leader_name,
                        price=0.0,  # 可通过 data_manager 获取实时价格
                        change_pct=sector.leader_change_pct,
                        sector=sector.name,
                        reason=f"{sector.name}龙头股，板块涨幅 {sector.change_pct:+.2f}%",
                        confidence=confidence
                    )
                    recommendations.append(stock)
        
        # 策略2：如果龙头股数据不足，可补充从热点板块中查找涨幅最大的个股
        # （此处需要对接数据源 API，暂时留作扩展）
        if len(recommendations) < max_stocks:
            logger.info(f"[个股推荐] 已推荐 {len(recommendations)} 支，可进一步补充")
        
        logger.info(f"[个股推荐] 完成推荐，共 {len(recommendations)} 支股票")
        return recommendations[:max_stocks]


def build_sector_capital_block(
    top_sectors: List[SectorWithCapital],
    bottom_sectors: List[SectorWithCapital],
    weights: Dict[str, float],
    language: str = "zh"
) -> str:
    """
    生成板块资金分析块
    
    包含资金量、权重、龙头股等信息
    """
    if not top_sectors and not bottom_sectors:
        return ""
    
    lines = []
    
    if language == "en":
        lines.extend([
            "#### Sector Capital & Leadership",
            "| Sector | Change | Capital (bn) | Weight | Leaders |",
            "|--------|--------|--------------|--------|---------|",
        ])
    else:
        lines.extend([
            "#### 板块资金分析",
            "| 板块 | 涨跌幅 | 成交额(亿) | 权重占比 | 龙头股 |",
            "|------|--------|----------|--------|--------|",
        ])
    
    # 领涨板块
    for rank, sector in enumerate(top_sectors[:5], 1):
        weight_str = f"{weights.get(sector.name, 0):.1f}%" if weights else "N/A"
        leader_str = f"{sector.leader_name}({sector.leader_change_pct:+.1f}%)" if sector.leader_name else "N/A"
        
        if language == "en":
            lines.append(
                f"| {rank}. {sector.name} | {sector.change_pct:+.2f}% | "
                f"{sector.amount:.0f} | {weight_str} | {leader_str} |"
            )
        else:
            lines.append(
                f"| {sector.name} | {sector.change_pct:+.2f}% | "
                f"{sector.amount:.0f} | {weight_str} | {leader_str} |"
            )
    
    return "\n".join(lines)


def build_stock_recommendation_block(
    recommendations: List[StockRecommendation],
    language: str = "zh"
) -> str:
    """
    生成个股推荐块
    """
    if not recommendations:
        return ""
    
    lines = []
    
    if language == "en":
        lines.extend([
            "#### Stock Recommendations (Top 5)",
            "| # | Code | Name | Sector | Reason | Confidence |",
            "|---|------|------|--------|--------|------------|",
        ])
    else:
        lines.extend([
            "#### 值得关注的个股 TOP 5",
            "| 序号 | 代码 | 名称 | 所属板块 | 推荐理由 | 信心度 |",
            "|------|------|------|--------|---------|--------|",
        ])
    
    for rank, stock in enumerate(recommendations[:5], 1):
        if language == "en":
            lines.append(
                f"| {rank} | {stock.code} | {stock.name} | {stock.sector} | "
                f"{stock.reason[:30]} | {stock.confidence:.0f}% |"
            )
        else:
            lines.append(
                f"| {rank} | {stock.code} | {stock.name} | {stock.sector} | "
                f"{stock.reason[:30]} | {stock.confidence:.0f}% |"
            )
    
    return "\n".join(lines)
