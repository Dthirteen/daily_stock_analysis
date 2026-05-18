# -*- coding: utf-8 -*-
"""
===================================
大盘复盘分析模块
===================================

职责：
1. 获取大盘指数数据（上证、深证、创业板）
2. 搜索市场新闻形成复盘情报
3. 使用大模型生成每日大盘复盘报告
4. 增强板块分析（资金量权重、推荐个股）
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List

import pandas as pd

from src.config import get_config
from src.report_language import normalize_report_language
from src.search_service import SearchService
from src.core.market_profile import get_profile, MarketProfile
from src.core.market_strategy import get_market_strategy_blueprint
from src.market_sector_analyzer import (
    MarketSectorAnalyzer, SectorWithCapital, StockRecommendation,
    build_sector_capital_block, build_stock_recommendation_block
)
from data_provider.base import DataFetcherManager

logger = logging.getLogger(__name__)


_ENGLISH_SECTION_PATTERNS = {
    "market_summary": r"###\s*(?:1\.\s*)?Market Summary",
    "index_commentary": r"###\s*(?:2\.\s*)?(?:Index Commentary|Major Indices)",
    "sector_highlights": r"###\s*(?:4\.\s*)?(?:Sector Highlights|Sector/Theme Highlights)",
    "stock_recommendations": r"###\s*(?:Stock Recommendations|Watchlist)",
}

_CHINESE_SECTION_PATTERNS = {
    "market_summary": r"###\s*一、(?:盘面总览|市场总结)",
    "index_commentary": r"###\s*二、(?:指数结构|指数点评|主要指数)",
    "sector_highlights": r"###\s*三、(?:板块主线|热点解读|板块表现)",
    "funds_sentiment": r"###\s*四、(?:资金与情绪|资金动向)",
    "stock_recommendations": r"###\s*(?:值得关注的个股|个股推荐|关注名单)",
    "news_catalysts": r"###\s*五、(?:消息催化|后市展望)",
}


@dataclass
class MarketIndex:
    """大盘指数数据"""
    code: str                    # 指数代码
    name: str                    # 指数名称
    current: float = 0.0         # 当前点位
    change: float = 0.0          # 涨跌点数
    change_pct: float = 0.0      # 涨跌幅(%)
    open: float = 0.0            # 开盘点位
    high: float = 0.0            # 最高点位
    low: float = 0.0             # 最低点位
    prev_close: float = 0.0      # 昨收点位
    volume: float = 0.0          # 成交量（手）
    amount: float = 0.0          # 成交额（元）
    amplitude: float = 0.0       # 振幅(%)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'name': self.name,
            'current': self.current,
            'change': self.change,
            'change_pct': self.change_pct,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'amount': self.amount,
            'amplitude': self.amplitude,
        }


@dataclass
class MarketOverview:
    """市场概览数据"""
    date: str                           # 日期
    indices: List[MarketIndex] = field(default_factory=list)  # 主要指数
    up_count: int = 0                   # 上涨家数
    down_count: int = 0                 # 下跌家数
    flat_count: int = 0                 # 平盘家数
    limit_up_count: int = 0             # 涨停家数
    limit_down_count: int = 0           # 跌停家数
    total_amount: float = 0.0           # 两市成交额（亿元）
    # north_flow: float = 0.0           # 北向资金净流入（亿元）- 已废弃，接口不可用

    # 板块涨幅榜
    top_sectors: List[Dict] = field(default_factory=list)     # 涨幅前5板块
    bottom_sectors: List[Dict] = field(default_factory=list)  # 跌幅前5板块

    # 增强板块数据
    top_sectors_enhanced: List[SectorWithCapital] = field(default_factory=list)  # 增强的领涨板块
    bottom_sectors_enhanced: List[SectorWithCapital] = field(default_factory=list)  # 增强的领跌板块
    sector_capital_weights: Dict[str, float] = field(default_factory=dict)  # 板块资金权重
    recommended_stocks: List[StockRecommendation] = field(default_factory=list)  # 推荐个股

    # ── 增强数据：资金流向 ──────────────────────────────
    capital_flow: Dict[str, Any] = field(default_factory=dict)          # {north_net_inflow, southbound_net_inflow, main_net_inflow, main_inflow_desc}
    sector_capital_flow_top: List[Dict] = field(default_factory=list)   # 净流入前N板块 [{name, net_inflow}]
    sector_capital_flow_bottom: List[Dict] = field(default_factory=list) # 净流出前N板块

    # ── 增强数据：市场宽度 ──────────────────────────────
    gain_distribution: Dict[str, int] = field(default_factory=dict)     # 涨幅区间分布 {">5%": count, ...}
    limit_up_detail: Dict[str, Any] = field(default_factory=dict)       # 涨停池明细 {total, first_limit, non_first_limit, broken_limit, broken_rate}
    yest_limit_avg_chg: float = 0.0                                    # 昨日涨停股今日平均涨跌幅(%)

    # ── 增强数据：指数技术面 ────────────────────────────
    index_technicals: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {index_code: {ma_status, macd_status, rsi_status, volume_status, score}}

    # ── 增强数据：板块持续性 ────────────────────────────
    persistent_leaders: List[str] = field(default_factory=list)   # 连续领涨板块名称列表
    new_leaders: List[str] = field(default_factory=list)         # 新晋领涨板块名称列表
    falling_leaders: List[str] = field(default_factory=list)     # 由涨转跌板块名称列表


class MarketAnalyzer:
    """
    大盘复盘分析器
    
    功能：
    1. 获取大盘指数实时行情
    2. 获取市场涨跌统计
    3. 获取板块涨跌榜（增强资金量、龙头股、推荐个股）
    4. 搜索市场新闻
    5. 生成大盘复盘报告
    """
    
    def __init__(
        self,
        search_service: Optional[SearchService] = None,
        analyzer=None,
        region: str = "cn",
    ):
        """
        初始化大盘分析器

        Args:
            search_service: 搜索服务实例
            analyzer: AI分析器实例（用于调用LLM）
            region: 市场区域 cn=A股 us=美股
        """
        self.config = get_config()
        self.search_service = search_service
        self.analyzer = analyzer
        self.data_manager = DataFetcherManager()
        self.region = region if region in ("cn", "us", "hk") else "cn"
        self.profile: MarketProfile = get_profile(self.region)
        self.strategy = get_market_strategy_blueprint(self.region)
        self.sector_analyzer = MarketSectorAnalyzer(self.data_manager)

    def _get_review_language(self) -> str:
        configured = normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )
        if self.region == "us":
            return "en"
        return configured

    def _get_template_review_language(self) -> str:
        return normalize_report_language(
            getattr(getattr(self, "config", None), "report_language", "zh")
        )

    def _get_market_scope_name(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "us":
            return "US market"
        if self.region == "hk":
            return "Hong Kong market" if review_language == "en" else "港股市场"
        if review_language == "en":
            return "A-share market"
        return "A股市场"

    def _get_turnover_unit_label(self) -> str:
        """Return the turnover unit label for the current market/language."""
        if self.region == "us":
            return "USD bn" if self._get_review_language() == "en" else "十亿美元"
        if self.region == "hk":
            return "HKD bn" if self._get_review_language() == "en" else "十亿港元"
        return "CNY 100m" if self._get_review_language() == "en" else "亿"

    def _format_turnover_value(self, amount_raw: float) -> str:
        """Format raw turnover according to market-specific units."""
        if amount_raw == 0.0:
            return "N/A"
        if self.region in ("us", "hk"):
            return f"{amount_raw / 1e9:.2f}"
        if amount_raw > 1e6:
            return f"{amount_raw / 1e8:.0f}"
        return f"{amount_raw:.0f}"

    def _get_review_title(self, date: str) -> str:
        if self._get_review_language() == "en":
            market_names = {"us": "US Market Recap", "hk": "HK Market Recap"}
            market_name = market_names.get(self.region, "A-share Market Recap")
            return f"## {date} {market_name}"
        return f"## {date} 大盘复盘"

    def _get_index_hint(self) -> str:
        if self._get_review_language() == "en":
            if self.region == "us":
                return "Analyze the key moves in the S&P 500, Nasdaq, Dow, and other major indices."
            if self.region == "hk":
                return "Analyze the key moves in the HSI, Hang Seng Tech, HSCEI, and other major indices."
            return "Analyze the price action in the SSE, SZSE, ChiNext, and other major indices."
        return self.profile.prompt_index_hint

    def _get_strategy_prompt_block(self) -> str:
        if self.region == "hk" and self._get_review_language() == "en":
            return """## Strategy Blueprint: Hong Kong Market Regime Strategy
Focus on HSI trend, southbound flow dynamics, and sector rotation to define next-session risk posture.

### Strategy Principles
- Read market regime from HSI, HSTECH, and HSCEI alignment first.
- Track southbound capital flow as a key sentiment driver.
- Translate recap into actionable risk-on/risk-off stance with clear invalidation points.

### Analysis Dimensions
- Trend Regime: Classify the market as momentum, range, or risk-off.
  - Are HSI/HSTECH/HSCEI directionally aligned
  - Did volume confirm the move
  - Are key index levels reclaimed or lost
- Capital Flows: Map southbound flow and macro narrative into equity risk appetite.
  - Southbound net flow direction and magnitude
  - USD/HKD and China policy implications
  - Breadth and leadership concentration
- Sector Themes: Identify persistent leaders and vulnerable laggards.
  - Tech/internet platform trend persistence
  - Financials/property sensitivity to policy shifts
  - Defensive vs growth factor rotation

### Action Framework
- Risk-on: broad index breakout with expanding southbound participation.
- Neutral: mixed index signals; focus on selective relative strength.
- Risk-off: failed breakouts and rising volatility; prioritize capital preservation."""
        if not (self.region == "cn" and self._get_review_language() == "en"):
            return self.strategy.to_prompt_block()
        return """## Strategy Blueprint: A-share Three-Phase Recap Strategy
Focus on index trend, liquidity, and sector rotation to shape the next-session trading plan.

### Strategy Principles
- Read index direction first, then confirm liquidity structure, and finally test sector persistence.
- Every conclusion must map to position sizing, trading pace, and risk-control actions.
- Base judgments on today's data and the latest 3-day news flow without inventing unverified information.

### Analysis Dimensions
- Trend Structure: Determine whether the market is in an uptrend, range, or defensive phase.
  - Are the SSE, SZSE, and ChiNext moving in the same direction
  - Is the market advancing on expanding volume or slipping on contracting volume
  - Have key support or resistance levels been reclaimed or broken
- Liquidity & Sentiment: Identify near-term risk appetite and market temperature.
  - Advance/decline breadth and limit-up/limit-down structure
  - Whether turnover is expanding or fading
  - Whether high-beta leaders are showing divergence
- Leading Themes: Distill tradable leadership and areas to avoid.
  - Whether leading sectors have clear event catalysts
  - Whether sector leaders are pulling the group higher
  - Whether weakness is broadening across lagging sectors

### Action Framework
- Offensive: indices rise in sync, turnover expands, and core themes strengthen.
- Balanced: index divergence or low-volume consolidation; keep sizing controlled and wait for confirmation.
- Defensive: indices weaken and laggards broaden; prioritize risk control and de-risking."""

    def _get_strategy_markdown_block(self, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if self.region == "hk" and review_language == "en":
            return """### 6. Strategy Framework
- **Trend Regime**: Classify the market as momentum, range, or risk-off based on HSI/HSTECH/HSCEI alignment.
- **Capital Flows**: Track southbound flow direction and macro narrative for risk appetite signals.
- **Sector Themes**: Focus on tech/internet platform persistence and financials/property policy sensitivity.
"""
        if not (self.region == "cn" and review_language == "en"):
            return self.strategy.to_markdown_block()
        return """### 6. Strategy Framework
- **Trend Structure**: Determine whether the market is in an uptrend, range, or defensive phase.
- **Liquidity & Sentiment**: Track breadth, turnover expansion, and whether leaders are diverging.
- **Leading Themes**: Focus on sectors with catalysts and sustained leadership while avoiding broadening weakness.
"""

    def _get_market_mood_text(self, mood_key: str, review_language: str | None = None) -> str:
        review_language = review_language or self._get_review_language()
        if review_language == "en":
            mapping = {
                "strong_up": "strong gains",
                "mild_up": "moderate gains",
                "mild_down": "mild losses",
                "strong_down": "clear weakness",
                "range": "range-bound trading",
            }
        else:
            mapping = {
                "strong_up": "强势上涨",
                "mild_up": "小幅上涨",
                "mild_down": "小幅下跌",
                "strong_down": "明显下跌",
                "range": "震荡整理",
            }
        return mapping[mood_key]

    def get_market_overview(self) -> MarketOverview:
        """
        获取市场概览数据
        
        Returns:
            MarketOverview: 市场概览数据对象
        """
        today = datetime.now().strftime('%Y-%m-%d')
        overview = MarketOverview(date=today)
        
        # 1. 获取主要指数行情（按 region 切换 A 股/美股）
        overview.indices = self._get_main_indices()

        # 2. 获取涨跌统计（A 股有，美股无等效数据）
        if self.profile.has_market_stats:
            self._get_market_statistics(overview)

        # 3. 获取板块涨跌榜（A 股有，美股暂无）
        if self.profile.has_sector_rankings:
            self._get_sector_rankings(overview)
        
        # 4. 增强板块分析（资金量权重、推荐个股）
        self._enhance_sector_analysis(overview)

        # 5. 获取北向/南向/主力资金流向（增强1）
        if self.profile.has_market_stats:
            self._get_capital_flow_data(overview)

        # 6. 获取市场宽度数据：涨幅分布 + 涨停池明细 + 昨日涨停表现（增强3）
        if self.profile.has_market_stats:
            self._get_market_breadth_data(overview)

        # 7. 指数技术面分析：MA/MACD/RSI/量价（增强2）
        self._get_index_technicals(overview)

        # 8. 板块持续性分析：对比昨日领涨板块（增强5）
        if self.profile.has_market_stats:
            self._analyze_sector_persistence(overview)

        # 9. 智能选股：综合多维度筛选7支建议关注的股票
        if self.profile.has_market_stats:
            self._get_smart_stock_picks(overview)
        
        # 9. 获取北向资金（可选）
        # self._get_north_flow(overview)
        
        return overview

    
    def _get_main_indices(self) -> List[MarketIndex]:
        """获取主要指数实时行情"""
        indices = []

        try:
            logger.info("[大盘] 获取主要指数实时行情...")

            # 使用 DataFetcherManager 获取指数行情（按 region 切换）
            data_list = self.data_manager.get_main_indices(region=self.region)

            if data_list:
                for item in data_list:
                    index = MarketIndex(
                        code=item['code'],
                        name=item['name'],
                        current=item['current'],
                        change=item['change'],
                        change_pct=item['change_pct'],
                        open=item['open'],
                        high=item['high'],
                        low=item['low'],
                        prev_close=item['prev_close'],
                        volume=item['volume'],
                        amount=item['amount'],
                        amplitude=item['amplitude']
                    )
                    indices.append(index)

            if not indices:
                logger.warning("[大盘] 所有行情数据源失败，将依赖新闻搜索进行分析")
            else:
                logger.info(f"[大盘] 获取到 {len(indices)} 个指数行情")

        except Exception as e:
            logger.error(f"[大盘] 获取指数行情失败: {e}")

        return indices

    def _get_market_statistics(self, overview: MarketOverview):
        """获取市场涨跌统计"""
        try:
            logger.info("[大盘] 获取市场涨跌统计...")

            stats = self.data_manager.get_market_stats()

            if stats:
                overview.up_count = stats.get('up_count', 0)
                overview.down_count = stats.get('down_count', 0)
                overview.flat_count = stats.get('flat_count', 0)
                overview.limit_up_count = stats.get('limit_up_count', 0)
                overview.limit_down_count = stats.get('limit_down_count', 0)
                overview.total_amount = stats.get('total_amount', 0.0)

                logger.info(f"[大盘] 涨:{overview.up_count} 跌:{overview.down_count} 平:{overview.flat_count} "
                          f"涨停:{overview.limit_up_count} 跌停:{overview.limit_down_count} "
                          f"成交额:{overview.total_amount:.0f}亿")

        except Exception as e:
            logger.error(f"[大盘] 获取涨跌统计失败: {e}")

    def _get_sector_rankings(self, overview: MarketOverview):
        """获取板块涨跌榜"""
        try:
            logger.info("[大盘] 获取板块涨跌榜...")

            top_sectors, bottom_sectors = self.data_manager.get_sector_rankings(5)

            if top_sectors or bottom_sectors:
                overview.top_sectors = top_sectors
                overview.bottom_sectors = bottom_sectors

                logger.info(f"[大盘] 领涨板块: {[s['name'] for s in overview.top_sectors]}")
                logger.info(f"[大盘] 领跌板块: {[s['name'] for s in overview.bottom_sectors]}")

        except Exception as e:
            logger.error(f"[大盘] 获取板块涨跌榜失败: {e}")

    def _enhance_sector_analysis(self, overview: MarketOverview):
        """
        增强板块分析：添加资金量、龙头股、推荐个股
        """
        try:
            logger.info("[大盘] 开始增强板块分析...")
            
            # 1. 获取增强的板块数据（包含资金量、龙头股）
            top_enhanced, bottom_enhanced = self.sector_analyzer.get_sector_rankings_with_capital(5)
            overview.top_sectors_enhanced = top_enhanced
            overview.bottom_sectors_enhanced = bottom_enhanced
            
            # 2. 计算板块资金权重
            weights = self.sector_analyzer.calculate_sector_capital_weights(top_enhanced, bottom_enhanced)
            overview.sector_capital_weights = weights
            
            # 3. 推荐值得关注的个股
            recommendations = self.sector_analyzer.get_recommended_stocks(top_enhanced, max_stocks=5)
            overview.recommended_stocks = recommendations
            
            logger.info(f"[大盘] 增强分析完成：推荐 {len(recommendations)} 支个股")
            
        except Exception as e:
            logger.error(f"[大盘] 增强板块分析失败: {e}")

    def _get_capital_flow_data(self, overview: MarketOverview):
        """获取资金流向数据（北向/南向/主力 + 行业资金流向排名）"""
        try:
            logger.info("[大盘] 获取资金流向数据...")

            # 1. 综合资金流向（北向/南向/主力）
            flow_data = self.data_manager.get_capital_flow()
            if flow_data:
                overview.capital_flow = flow_data
                logger.info(
                    f"[大盘] 资金流向: 北向={flow_data.get('north_net_inflow', 'N/A')}亿, "
                    f"主力={flow_data.get('main_net_inflow', 'N/A')}亿({flow_data.get('main_inflow_desc', '')})"
                )

            # 2. 行业板块资金流向排名
            sector_top, sector_bottom = self.data_manager.get_sector_capital_flow(5)
            if sector_top or sector_bottom:
                overview.sector_capital_flow_top = sector_top
                overview.sector_capital_flow_bottom = sector_bottom
                logger.info(f"[大盘] 板块资金流向: 流入TOP {[s['name'] for s in sector_top[:3]]}")

        except Exception as e:
            logger.error(f"[大盘] 获取资金流向失败: {e}")

    def _get_market_breadth_data(self, overview: MarketOverview):
        """获取市场宽度数据（涨幅分布 + 涨停池明细 + 昨日涨停表现）"""
        try:
            logger.info("[大盘] 获取市场宽度数据...")

            breadth = self.data_manager.get_market_breadth()
            if not breadth:
                return

            if 'gain_distribution' in breadth:
                overview.gain_distribution = breadth['gain_distribution']
                dist = breadth['gain_distribution']
                logger.info(f"[大盘] 涨幅分布: {dist}")

            if 'limit_up_detail' in breadth:
                overview.limit_up_detail = breadth['limit_up_detail']
                detail = breadth['limit_up_detail']
                logger.info(
                    f"[大盘] 涨停池: 总计{detail.get('total', 0)}家, "
                    f"一字板{detail.get('first_limit', 0)}, 炸板{detail.get('broken_limit', 0)}({detail.get('broken_rate', 0)}%)"
                )

            if 'yest_limit_avg_chg' in breadth:
                overview.yest_limit_avg_chg = breadth['yest_limit_avg_chg']
                logger.info(f"[大盘] 昨日涨停今日均涨幅: {breadth['yest_limit_avg_chg']}%")

        except Exception as e:
            logger.error(f"[大盘] 获取市场宽度失败: {e}")

    def _get_index_technicals(self, overview: MarketOverview):
        """获取主要指数的技术面分析（MA排列/MACD/RSI/量价）"""
        if not self.profile.has_market_stats:
            # 美股/港股暂不计算指数技术面（可后续扩展）
            return

        try:
            logger.info("[大盘] 计算指数技术面指标...")

            # 分析前几个主要指数
            index_codes_to_analyze = []
            for idx in overview.indices[:4]:
                code = idx.code
                if code and (code.startswith('sh') or code.startswith('sz')):
                    index_codes_to_analyze.append(code)

            technicals = {}
            for code in index_codes_to_analyze:
                df = self.data_manager.get_index_daily_history(code, days=60)
                if df is None or len(df) < 20:
                    continue

                tech = self._analyze_single_index_technical(df, code)
                if tech:
                    technicals[code] = tech

            if technicals:
                overview.index_technicals = technicals
                logger.info(f"[大盘] 指数技术面分析完成: {list(technicals.keys())}")

        except Exception as e:
            logger.error(f"[大盘] 指数技术面分析失败: {e}")

    def _analyze_single_index_technical(self, df: pd.DataFrame, index_code: str) -> Optional[Dict[str, Any]]:
        """对单只指数进行轻量级技术分析"""
        import numpy as np

        result = {}
        try:
            close = pd.to_numeric(df['close'], errors='coerce')
            volume = pd.to_numeric(df['volume'], errors='coerce')
            high = pd.to_numeric(df['high'], errors='coerce')
            low = pd.to_numeric(df['low'], errors='coerce')

            if len(close) < 10:
                return None

            latest = close.iloc[-1]
            prev_close = close.iloc[-2]

            # ── MA 排列状态 ──
            ma5 = close.rolling(5).mean().iloc[-1]
            ma10 = close.rolling(10).mean().iloc[-1]
            ma20 = close.rolling(20).mean().iloc[-1]
            ma60 = close.rolling(min(60, len(close))).mean().iloc[-1]

            result['ma5'] = round(float(ma5), 2)
            result['ma10'] = round(float(ma10), 2)
            result['ma20'] = round(float(ma20), 2)

            if pd.notna(ma60):
                result['ma60'] = round(float(ma60), 2)

            # 判断均线排列
            if all(pd.notna(x) for x in [ma5, ma10, ma20]):
                if ma5 > ma10 > ma20:
                    result['ma_status'] = "多头排列"
                    ma_score = 100
                elif ma5 < ma10 < ma20:
                    result['ma_status'] = "空头排列"
                    ma_score = 0
                else:
                    result['ma_status'] = "纠缠震荡"
                    ma_score = 50
            else:
                result['ma_status'] = "数据不足"
                ma_score = 50

            # ── MACD 状态 ──
            ema12 = close.ewm(span=12).mean()
            ema26 = close.ewm(span=26).mean()
            dif = ema12 - ema26
            dea = dif.ewm(span=9).mean()
            macd_bar = (dif - dea) * 2

            current_dif = float(dif.iloc[-1]) if pd.notna(dif.iloc[-1]) else 0
            prev_dif = float(dif.iloc[-2]) if len(dif) >= 2 and pd.notna(dif.iloc[-2]) else 0
            current_macd = float(macd_bar.iloc[-1]) if pd.notna(macd_bar.iloc[-1]) else 0

            result['macd_dif'] = round(current_dif, 4)
            result['macd_hist'] = round(current_macd, 4)

            if current_dif > 0 and prev_dif <= 0:
                result['macd_status'] = "金叉"
                macd_score = 90
            elif current_dif < 0 and prev_dif >= 0:
                result['macd_status'] = "死叉"
                macd_score = 10
            elif current_dif > 0:
                result['macd_status'] = "多头区域"
                macd_score = 70
            elif current_dif < 0:
                result['macd_status'] = "空头区域"
                macd_score = 30
            else:
                result['macd_status'] = "零轴附近"
                macd_score = 50

            # ── RSI 状态 ──
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            current_rsi = float(rsi.iloc[-1]) if pd.notna(rsi.iloc[-1]) else 50
            result['rsi'] = round(current_rsi, 1)

            if current_rsi > 70:
                result['rsi_status'] = "超买区"
                rsi_score = 30  # 超买意味着回调风险
            elif current_rsi < 30:
                result['rsi_status'] = "超卖区"
                rsi_score = 80  # 超卖可能反弹
            elif current_rsi > 55:
                result['rsi_status'] = "偏强"
                rsi_score = 70
            elif current_rsi < 45:
                result['rsi_status'] = "偏弱"
                rsi_score = 35
            else:
                result['rsi_status'] = "中性"
                rsi_score = 50

            # ── 量能状态 ──
            vol_ma5 = volume.rolling(5).mean().iloc[-1]
            vol_today = float(volume.iloc[-1]) if pd.notna(volume.iloc[-1]) else 0
            vol_ratio = vol_today / vol_ma5 if vol_ma5 and vol_ma5 > 0 else 1.0

            result['vol_today'] = int(vol_today)
            result['vol_ma5'] = int(vol_ma5)
            result['vol_ratio'] = round(vol_ratio, 2)

            if vol_ratio > 1.3:
                result['volume_status'] = "明显放量"
                vol_score = 85
            elif vol_ratio > 1.05:
                result['volume_status'] = "温和放量"
                vol_score = 70
            elif vol_ratio < 0.75:
                result['volume_status'] = "明显缩量"
                vol_score = 30
            elif vol_ratio < 0.95:
                result['volume_status'] = "温和缩量"
                vol_score = 45
            else:
                result['volume_status'] = "量能持平"
                vol_score = 55

            # ── 综合评分（加权）──
            total_score = int(round(ma_score * 0.30 + macd_score * 0.25 + rsi_score * 0.20 + vol_score * 0.25))
            result['score'] = max(0, min(100, total_score))

            return result

        except Exception as e:
            logger.warning(f"[大盘] 指数 {index_code} 技术面计算异常: {e}")
            return None

    def _analyze_sector_persistence(self, overview: MarketOverview):
        """
        分析板块持续性：对比今日与昨日领涨板块，识别连续领涨/新晋/由涨转跌板块
        通过保存和读取昨日板块数据实现跨日对比
        """
        import json
        import os

        try:
            today_top_names = set(s.get('name', '') for s in overview.top_sectors)
            if not today_top_names:
                return

            cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
            cache_file = os.path.join(cache_dir, 'market_yesterday_sectors.json')

            yesterday_top_names = set()

            # 尝试读取昨日缓存
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cached = json.load(f)
                        yesterday_top_names = set(cached.get('top_sectors', []))
                except Exception as e:
                    logger.debug(f"[大盘] 读取昨日板块缓存失败: {e}")

            # 计算交集和差集
            persistent = sorted(today_top_names & yesterday_top_names)
            new_leaders = sorted(today_top_names - yesterday_top_names)
            falling_leaders = sorted(yesterday_top_names - today_top_names)

            overview.persistent_leaders = persistent
            overview.new_leaders = new_leaders
            overview.falling_leaders = falling_leaders

            logger.info(
                f"[大盘] 板块持续性: 连续领涨={persistent}, 新晋={new_leaders}, 由涨转跌={falling_leaders}"
            )

            # 缓存今日数据供明日使用
            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'date': overview.date,
                        'top_sectors': list(today_top_names),
                        'bottom_sectors': [s.get('name', '') for s in overview.bottom_sectors],
                    }, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.debug(f"[大盘] 写入板块缓存失败: {e}")

        except Exception as e:
            logger.error(f"[大盘] 板块持续性分析失败: {e}")

    def _get_smart_stock_picks(self, overview: MarketOverview):
        """
        智能选股：综合多维度筛选 7 支建议关注的股票

        选股策略（优先级从高到低）：
        1. 涨停池强势股（非一字板，有换手，说明资金认可）
        2. 连续领涨板块的龙头股（板块持续性强 → 龙头确定性高）
        3. 新晋领涨板块的放量龙头（新热点启动信号）
        4. 资金大幅流入板块的领头股

        综合评分 = 涨跌幅权重(25) + 板块热度(25) + 换手率(20) + 成交额(15) + 资金面(15)
        """
        import pandas as pd

        if not self.profile.has_market_stats:
            return

        try:
            logger.info("[大盘] 开始智能选股，目标 7 支...")
            all_picks: Dict[str, StockRecommendation] = {}  # code -> recommendation (去重)

            # ── 策略1：涨停池强势股（非一字板优先，说明可参与）──
            limit_up_stocks = self.data_manager.get_limit_up_pool_stocks()
            if limit_up_stocks:
                # 过滤掉一字板（无法买入），按涨跌幅排序取前3
                tradable = [s for s in limit_up_stocks if not s.get('reason', '').startswith('一字')]
                if not tradable:
                    tradable = limit_up_stocks  # 全是一字板时也纳入

                for s in sorted(tradable, key=lambda x: x.get('change_pct', 0), reverse=True)[:3]:
                    rec = StockRecommendation(
                        code=s['code'],
                        name=s['name'],
                        change_pct=s.get('change_pct', 0.0),
                        reason=f"涨停{'('+s['reason']+')' if s.get('reason') else ''}",
                        confidence=min(95, 75 + s.get('change_pct', 0) * 0.5),
                        pick_source='limit_up',
                    )
                    all_picks[s['code']] = rec
                logger.info(f"[选股] 策略1(涨停池): 已选 {min(3, len(tradable))} 支")

            # ── 策略2：连续领涨板块龙头 ──
            if overview.persistent_leaders:
                for sector_name in overview.persistent_leaders[:2]:
                    df = self.data_manager.get_sector_constituents(sector_name)
                    if df is not None and not df.empty:
                        # 取涨幅最大的作为龙头
                        chg_col = 'change_pct' if 'change_pct' in df.columns else '涨跌幅'
                        if chg_col in df.columns:
                            df[chg_col] = pd.to_numeric(df[chg_col], errors='coerce')
                            leader = df.nlargest(1, chg_col).iloc[0]
                            code = str(leader.get('code', leader.get('代码', '')))
                            if code and code not in all_picks:
                                name = str(leader.get('name', leader.get('名称', '')))
                                chg = float(leader[chg_col]) if pd.notna(leader[chg_col]) else 0.0
                                turnover = float(leader.get('turnover_rate', leader.get('换手率', 0)))
                                amount = float(leader.get('amount', leader.get('成交额', 0)))
                                rec = StockRecommendation(
                                    code=code,
                                    name=name,
                                    change_pct=chg,
                                    sector=sector_name,
                                    reason=f"{sector_name}连续领涨龙头",
                                    confidence=min(90, 70 + abs(chg) * 0.5),
                                    pick_source='sector_leader',
                                    turnover_rate=turnover,
                                    amount=amount / 1e8 if amount > 1e6 else amount,
                                )
                                all_picks[code] = rec
                logger.info(f"[选股] 策略2(连续领涨龙头): 已选若干支")

            # ── 策略3：今日新晋热门板块龙头 ──
            if overview.new_leaders:
                for sector_name in overview.new_leaders[:2]:
                    df = self.data_manager.get_sector_constituents(sector_name)
                    if df is not None and not df.empty:
                        chg_col = 'change_pct' if 'change_pct' in df.columns else '涨跌幅'
                        if chg_col in df.columns:
                            df[chg_col] = pd.to_numeric(df[chg_col], errors='coerce')
                            # 优先选涨幅大且换手高的（说明资金大量介入）
                            turnover_col = 'turnover_rate' if 'turnover_rate' in df.columns else ('换手率' if '换手率' in df.columns else None)
                            if turnover_col:
                                df[turnover_col] = pd.to_numeric(df[turnover_col], errors='coerce')
                                leader = df.iloc[(df[chg_col] * 0.6 + df.get(turnover_col, 0).fillna(0) * 10).nlargest(1).index[0]]
                            else:
                                leader = df.nlargest(1, chg_col).iloc[0]

                            code = str(leader.get('code', leader.get('代码', '')))
                            if code and code not in all_picks:
                                name = str(leader.get('name', leader.get('名称', '')))
                                chg = float(leader[chg_col]) if pd.notna(leader[chg_col]) else 0.0
                                turnover = float(leader.get(turnover_col or 'turnover_rate', 0))
                                amount = float(leader.get('amount', leader.get('成交额', 0)))
                                rec = StockRecommendation(
                                    code=code,
                                    name=name,
                                    change_pct=chg,
                                    sector=sector_name,
                                    reason=f"{sector_name}新晋热点，放量启动",
                                    confidence=min(85, 65 + abs(chg) * 0.5 + min(turnover * 0.5, 10)),
                                    pick_source='hot_sector',
                                    turnover_rate=turnover,
                                    amount=amount / 1e8 if amount > 1e6 else amount,
                                )
                                all_picks[code] = rec
                logger.info(f"[选股] 策略3(新晋热点龙头): 已选若干支")

            # ── 策略4：资金流入板块领头股补充 ──
            if overview.sector_capital_flow_top and len(all_picks) < 7:
                for s in overview.sector_capital_flow_top[:2]:
                    sector_name = s.get('name', '')
                    if not sector_name:
                        continue
                    df = self.data_manager.get_sector_constituents(sector_name)
                    if df is not None and not df.empty:
                        chg_col = 'change_pct' if 'change_pct' in df.columns else '涨跌幅'
                        if chg_col in df.columns:
                            df[chg_col] = pd.to_numeric(df[chg_col], errors='coerce')
                            leader = df.nlargest(1, chg_col).iloc[0]
                            code = str(leader.get('code', leader.get('代码', '')))
                            if code and code not in all_picks:
                                name = str(leader.get('name', leader.get('名称', '')))
                                chg = float(leader[chg_col]) if pd.notna(leader[chg_col]) else 0.0
                                inflow = s.get('net_inflow', 0)
                                rec = StockRecommendation(
                                    code=code,
                                    name=name,
                                    change_pct=chg,
                                    sector=sector_name,
                                    reason=f"{sector_name}资金净流入{inflow:+.1f}亿龙头",
                                    confidence=min(80, 60 + min(inflow, 20)),
                                    pick_source='capital_inflow',
                                )
                                all_picks[code] = rec
                                if len(all_picks) >= 7:
                                    break
                logger.info(f"[选股] 策略4(资金流入龙头): 补充完毕")

            # ── 排序 & 截断到 7 支 ──
            final_picks = sorted(
                all_picks.values(),
                key=lambda x: x.confidence,
                reverse=True
            )[:7]

            # 重新计算排名后的信心度（让分值更有区分度）
            for i, pick in enumerate(final_picks):
                pick.confidence = max(40, min(98, 95 - i * 8))

            overview.recommended_stocks = final_picks
            logger.info(f"[大盘] 智能选股完成：共 {len(final_picks)} 支")
            for p in final_picks:
                logger.info(f"  [{p.pick_source}] {p.code} {p.name}: {p.reason} | 信心度:{p.confidence:.0f}")

        except Exception as e:
            logger.error(f"[大盘] 智能选股失败: {e}")
            # fallback 到原有逻辑
            if not overview.recommended_stocks and overview.top_sectors_enhanced:
                overview.recommended_stocks = self.sector_analyzer.get_recommended_stocks(
                    overview.top_sectors_enhanced, max_stocks=7
                )

    # def _get_north_flow(self, overview: MarketOverview):
    #     """获取北向资金流入"""
    #     try:
    #         logger.info("[大盘] 获取北向资金...")
    #         
    #         # 获取北向资金数据
    #         df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
    #         
    #         if df is not None and not df.empty:
    #             # 取最新一条数据
    #             latest = df.iloc[-1]
    #             if '当日净流入' in df.columns:
    #                 overview.north_flow = float(latest['当日净流入']) / 1e8  # 转为亿元
    #             elif '净流入' in df.columns:
    #                 overview.north_flow = float(latest['净流入']) / 1e8
    #                 
    #             logger.info(f"[大盘] 北向资金净流入: {overview.north_flow:.2f}亿")
    #             
    #     except Exception as e:
    #         logger.warning(f"[大盘] 获取北向资金失败: {e}")
    
    def search_market_news(self) -> List[Dict]:
        """
        搜索市场新闻
        
        Returns:
            新闻列表
        """
        if not self.search_service:
            logger.warning("[大盘] 搜索服务未配置，跳过新闻搜索")
            return []
        
        all_news = []

        # 按 region 使用不同的新闻搜索词
        search_queries = self.profile.news_queries
        
        try:
            logger.info("[大盘] 开始搜索市场新闻...")
            
            # 根据 region 设置搜索上下文名称，避免美股搜索被解读为 A 股语境
            market_names = {"cn": "大盘", "us": "US market", "hk": "HK market"}
            market_name = market_names.get(self.region, "大盘")
            for query in search_queries:
                response = self.search_service.search_stock_news(
                    stock_code="market",
                    stock_name=market_name,
                    max_results=3,
                    focus_keywords=query.split()
                )
                if response and response.results:
                    all_news.extend(response.results)
                    logger.info(f"[大盘] 搜索 '{query}' 获取 {len(response.results)} 条结果")
            
            logger.info(f"[大盘] 共获取 {len(all_news)} 条市场新闻")
            
        except Exception as e:
            logger.error(f"[大盘] 搜索市场新闻失败: {e}")
        
        return all_news
    
    def generate_market_review(self, overview: MarketOverview, news: List) -> str:
        """
        使用大模型生成大盘复盘报告
        
        Args:
            overview: 市场概览数据
            news: 市场新闻列表 (SearchResult 对象列表)
            
        Returns:
            大盘复盘报告文本
        """
        if not self.analyzer or not self.analyzer.is_available():
            logger.warning("[大盘] AI分析器未配置或不可用，使用模板生成报告")
            return self._generate_template_review(overview, news)
        
        # 构建 Prompt
        prompt = self._build_review_prompt(overview, news)
        
        logger.info("[大盘] 调用大模型生成复盘报告...")
        # Use the public generate_text() entry point — never access private analyzer attributes.
        review = self.analyzer.generate_text(prompt, max_tokens=8192, temperature=0.7)

        if review:
            logger.info("[大盘] 复盘报告生成成功，长度: %d 字符", len(review))
            # Inject structured data tables into LLM prose sections
            return self._inject_data_into_review(review, overview, news)
        else:
            logger.warning("[大盘] 大模型返回为空，使用模板报告")
            return self._generate_template_review(overview, news)
    
    def _inject_data_into_review(
        self,
        review: str,
        overview: MarketOverview,
        news: Optional[List] = None,
    ) -> str:
        """Inject structured data tables into the corresponding LLM prose sections."""
        # Build data blocks
        stats_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        sector_capital_block = self._build_sector_capital_block(overview)
        stock_recommendation_block = self._build_stock_recommendation_block(overview)
        news_block = self._build_news_block(news or [])
        # ── 增强数据块 ──
        capital_flow_block = self._build_capital_flow_block(overview)
        market_breadth_block = self._build_market_breadth_block(overview)
        index_technicals_block = self._build_index_technicals_block(overview)
        sector_persistence_block = self._build_sector_persistence_block(overview)

        patterns = (
            _ENGLISH_SECTION_PATTERNS
            if self._get_review_language() == "en"
            else _CHINESE_SECTION_PATTERNS
        )

        if stats_block:
            review = self._insert_after_section(
                review, patterns["market_summary"], stats_block,
            )

        if indices_block:
            review = self._insert_after_section(
                review, patterns["index_commentary"], indices_block,
            )

        if sector_capital_block:
            review = self._insert_after_section(
                review, patterns["sector_highlights"], sector_capital_block,
            )

        if stock_recommendation_block and "stock_recommendations" in patterns:
            review = self._insert_after_section(
                review, patterns["stock_recommendations"], stock_recommendation_block,
            )

        # ── 注入增强版选股表格（覆盖/追加到个股段落）──
        smart_picks_block = self._build_stock_recommendation_block(overview)
        if smart_picks_block:
            # 尝试注入到 Stock Recommendations / 值得关注个股 段落
            review = self._insert_after_section(
                review,
                patterns.get("stock_recommendations", r"###\s*(?:5\.\s*)?(?:Stock Recommendations|值得关注的个股)"),
                smart_picks_block,
            )

        if news_block and "news_catalysts" in patterns:
            review = self._insert_after_section(
                review, patterns["news_catalysts"], news_block,
            )

        # ── 注入增强数据块 ──
        # 资金流向注入到 Fund Flows / 资金与情绪 段落
        if capital_flow_block:
            review = self._insert_after_section(
                review,
                _ENGLISH_SECTION_PATTERNS.get("fund_flows", r"###\s*(?:3\.\s*)?(?:Fund Flows|Capital)")
                if self._get_review_language() == "en"
                else _CHINESE_SECTION_PATTERNS.get("funds_sentiment", r"###\s*五、(?:资金与情绪|资金动向)"),
                capital_flow_block,
            )

        # 市场宽度注入到资金与情绪段落后（或作为独立段落）
        if market_breadth_block:
            review = self._insert_after_section(
                review,
                _ENGLISH_SECTION_PATTERNS.get("fund_flows", r"###\s*(?:3\.\s*)?(?:Fund Flows|Capital)")
                if self._get_review_language() == "en"
                else _CHINESE_SECTION_PATTERNS.get("funds_sentiment", r"###\s*五、(?:资金与情绪|资金动向)"),
                market_breadth_block,
            )

        # 指数技术面注入到指数结构段落后
        if index_technicals_block:
            review = self._insert_after_section(
                review, patterns["index_commentary"], index_technicals_block,
            )

        # 板块持续性注入到板块主线段落后
        if sector_persistence_block:
            review = self._insert_after_section(
                review, patterns["sector_highlights"], sector_persistence_block,
            )

        return review

    @staticmethod
    def _insert_after_section(text: str, heading_pattern: str, block: str) -> str:
        """Insert a data block at the end of a markdown section (before the next ### heading)."""
        import re
        # Find the heading
        match = re.search(heading_pattern, text)
        if not match:
            return text
        start = match.end()
        # Find the next ### heading after this one
        next_heading = re.search(r'\n###\s', text[start:])
        if next_heading:
            insert_pos = start + next_heading.start()
        else:
            # No next heading — append at end
            insert_pos = len(text)
        # Insert the block before the next heading, with spacing
        return text[:insert_pos].rstrip() + '\n\n' + block + '\n\n' + text[insert_pos:].lstrip('\n')

    def _build_stats_block(self, overview: MarketOverview) -> str:
        """Build market statistics block."""
        has_stats = overview.up_count or overview.down_count or overview.total_amount
        if not has_stats:
            return ""
        if self._get_review_language() == "en":
            return (
                f"> 📈 Advancers **{overview.up_count}** / Decliners **{overview.down_count}** / "
                f"Flat **{overview.flat_count}** | "
                f"Limit-up **{overview.limit_up_count}** / Limit-down **{overview.limit_down_count}** | "
                f"Turnover **{overview.total_amount:.0f}** ({self._get_turnover_unit_label()})"
            )
        score, label = self._build_market_temperature(overview)
        participation = overview.up_count + overview.down_count + overview.flat_count
        up_ratio = overview.up_count / participation if participation else 0.0
        limit_spread = overview.limit_up_count - overview.limit_down_count
        lines = [
            f"> **盘面温度**：{label} **{score}/100** {self._build_temperature_bar(score)}",
            "",
            "| 指标 | 数值 | 观察 |",
            "|------|------|------|",
            f"| 上涨/下跌/平盘 | {overview.up_count} / {overview.down_count} / {overview.flat_count} | 上涨占比 {up_ratio:.1%} |",
            f"| 涨停/跌停 | {overview.limit_up_count} / {overview.limit_down_count} | 涨跌停差 {limit_spread:+d} |",
            f"| 两市成交额 | {overview.total_amount:.0f} 亿 | {self._describe_turnover(overview.total_amount)} |",
        ]
        return "\n".join(lines)

    def _build_indices_block(self, overview: MarketOverview) -> str:
        """构建指数行情表格"""
        if not overview.indices:
            return ""
        if self._get_review_language() == "en":
            lines = [
                f"| Index | Last | Change % | Open | High | Low | Amplitude | Turnover ({self._get_turnover_unit_label()}) |",
                "|-------|------|----------|------|------|-----|-----------|-----------------|",
            ]
        else:
            lines = [
                "| 指数 | 最新 | 涨跌幅 | 开盘 | 最高 | 最低 | 振幅 | 成交额(亿) |",
                "|------|------|--------|------|------|------|------|-----------|",
            ]
        for idx in overview.indices:
            arrow = "🔴" if idx.change_pct < 0 else "🟢" if idx.change_pct > 0 else "⚪"
            amount_raw = idx.amount or 0.0
            amount_str = self._format_turnover_value(amount_raw)
            lines.append(
                f"| {idx.name} | {idx.current:.2f} | {arrow} {idx.change_pct:+.2f}% | "
                f"{self._format_optional_number(idx.open)} | {self._format_optional_number(idx.high)} | "
                f"{self._format_optional_number(idx.low)} | {self._format_optional_pct(idx.amplitude)} | {amount_str} |"
            )
        return "\n".join(lines)

    def _build_sector_block(self, overview: MarketOverview) -> str:
        """Build sector ranking block."""
        if not overview.top_sectors and not overview.bottom_sectors:
            return ""
        lines = []
        if overview.top_sectors:
            if self._get_review_language() == "en":
                lines.extend([
                    "#### Leading Sectors",
                    "| Rank | Sector | Change |",
                    "|------|--------|--------|",
                ])
            else:
                lines.extend([
                    "#### 领涨板块 Top 5",
                    "| 排名 | 板块 | 涨跌幅 |",
                    "|------|------|--------|",
                ])
            for rank, sector in enumerate(overview.top_sectors[:5], 1):
                lines.append(
                    f"| {rank} | {sector.get('name', '-')} | {self._format_signed_pct(sector.get('change_pct'))} |"
                )
        if overview.bottom_sectors:
            if lines:
                lines.append("")
            if self._get_review_language() == "en":
                lines.extend([
                    "#### Lagging Sectors",
                    "| Rank | Sector | Change |",
                    "|------|--------|--------|",
                ])
            else:
                lines.extend([
                    "#### 领跌板块 Top 5",
                    "| 排名 | 板块 | 涨跌幅 |",
                    "|------|------|--------|",
                ])
            for rank, sector in enumerate(overview.bottom_sectors[:5], 1):
                lines.append(
                    f"| {rank} | {sector.get('name', '-')} | {self._format_signed_pct(sector.get('change_pct'))} |"
                )
        return "\n".join(lines)

    def _build_sector_capital_block(self, overview: MarketOverview) -> str:
        """Build sector capital analysis block with weights and leaders."""
        if not overview.top_sectors_enhanced and not overview.bottom_sectors_enhanced:
            return ""
        return build_sector_capital_block(
            overview.top_sectors_enhanced,
            overview.bottom_sectors_enhanced,
            overview.sector_capital_weights,
            language=self._get_review_language()
        )

    def _build_stock_recommendation_block(self, overview: MarketOverview) -> str:
        """Build stock recommendation block (enhanced: 7 stocks with source tags)."""
        if not overview.recommended_stocks:
            return ""

        lang = self._get_review_language()
        lines = []

        # 来源标签映射
        source_tags = {
            'limit_up': '🔥涨停',
            'sector_leader': '🏆领涨',
            'hot_sector': '🆕热点',
            'capital_inflow': '💰资金',
            '': '⭐推荐',
        }

        if lang == "en":
            lines.extend([
                "#### 🎯 Suggested Stock Picks (Top 7)",
                "| # | Code | Name | Chg% | Sector | Source | Rationale | Confidence |",
                "|---|------|------|-----|--------|--------|-----------|------------|",
            ])
            for rank, stock in enumerate(overview.recommended_stocks[:7], 1):
                src = source_tags.get(stock.pick_source, '⭐')
                reason_short = stock.reason[:20] if len(stock.reason) > 20 else stock.reason
                lines.append(
                    f"| {rank} | {stock.code} | {stock.name} | "
                    f"{stock.change_pct:+.1f}% | {stock.sector or '-'} | {src} | "
                    f"{reason_short} | {stock.confidence:.0f}% |"
                )
        else:
            lines.extend([
                "#### 🎯 建议关注个股 TOP 7",
                "| 序号 | 代码 | 名称 | 涨跌幅 | 所属板块 | 来源 | 推荐理由 | 信心度 |",
                "|------|------|------|--------|---------|------|---------|--------|",
            ])
            for rank, stock in enumerate(overview.recommended_stocks[:7], 1):
                src = source_tags.get(stock.pick_source, '⭐')
                reason_short = stock.reason[:20] if len(stock.reason) > 20 else stock.reason
                lines.append(
                    f"| {rank} | {stock.code} | {stock.name} | "
                    f"{stock.change_pct:+.1f}% | {stock.sector or '-'} | {src} | "
                    f"{reason_short} | {stock.confidence:.0f}% |"
                )

        return "\n".join(lines)

    def _build_news_block(self, news: List) -> str:
        """Build a compact news catalyst table for the rendered report."""
        if not news:
            return ""
        if self._get_review_language() == "en":
            lines = [
                "#### News Catalysts",
                "| # | Headline | Signal |",
                "|---|----------|--------|",
            ]
        else:
            lines = [
                "#### 近三日催化线索",
                "| 序号 | 事件/标题 | 关注点 |",
                "|------|-----------|--------|",
            ]

        for idx, item in enumerate(news[:5], 1):
            if hasattr(item, "title"):
                title = getattr(item, "title", "") or "-"
                snippet = getattr(item, "snippet", "") or ""
            else:
                title = item.get("title", "-") or "-"
                snippet = item.get("snippet", "") or ""
            title = self._escape_table_cell(str(title).strip()[:42])
            signal = self._escape_table_cell(str(snippet).strip().replace("\n", " ")[:58] or "-")
            lines.append(f"| {idx} | {title} | {signal} |")
        return "\n".join(lines)

    # ── 增强数据表格构建方法 ──────────────────────────────

    def _build_capital_flow_block(self, overview: MarketOverview) -> str:
        """构建资金流向表格（北向/南向/主力 + 行业资金流向排名）"""
        if not overview.capital_flow and not overview.sector_capital_flow_top:
            return ""

        lang = self._get_review_language()
        lines = []

        if lang == "en":
            lines.append("#### Capital Flows")
        else:
            lines.append("#### 资金流向")

        # 综合资金流向
        cf = overview.capital_flow
        if cf:
            if lang == "en":
                lines.append("| Flow Type | Net Inflow (100M) | Signal |")
                lines.append("|----------|-------------------|--------|")
                north = cf.get('north_net_inflow')
                south = cf.get('southbound_net_inflow')
                main_val = cf.get('main_net_inflow')
                main_desc = cf.get('main_inflow_desc', '')
                if north is not None:
                    arrow = "🟢" if north > 0 else "🔴" if north < 0 else "⚪"
                    lines.append(f"| Northbound | {arrow} {north:+.2f} | {'Inflow' if north > 0 else 'Outflow' if north < 0 else 'Flat'} |")
                if south is not None:
                    arrow = "🟢" if south > 0 else "🔴" if south < 0 else "⚪"
                    lines.append(f"| Southbound | {arrow} {south:+.2f} | {'Inflow' if south > 0 else 'Outflow' if south < 0 else 'Flat'} |")
                if main_val is not None:
                    arrow = "🟢" if main_val > 0 else "🔴" if main_val < 0 else "⚪"
                    lines.append(f"| Institutional | {arrow} {main_val:+.2f} | {main_desc} |")
            else:
                lines.append("| 资金类型 | 净流入(亿) | 信号 |")
                lines.append("|----------|-----------|------|")
                north = cf.get('north_net_inflow')
                south = cf.get('southbound_net_inflow')
                main_val = cf.get('main_net_inflow')
                main_desc = cf.get('main_inflow_desc', '')
                if north is not None:
                    arrow = "🟢" if north > 0 else "🔴" if north < 0 else "⚪"
                    lines.append(f"| 北向资金 | {arrow} {north:+.2f} | {'净流入' if north > 0 else '净流出' if north < 0 else '持平'} |")
                if south is not None:
                    arrow = "🟢" if south > 0 else "🔴" if south < 0 else "⚪"
                    lines.append(f"| 南向资金 | {arrow} {south:+.2f} | {'净流入' if south > 0 else '净流出' if south < 0 else '持平'} |")
                if main_val is not None:
                    arrow = "🟢" if main_val > 0 else "🔴" if main_val < 0 else "⚪"
                    lines.append(f"| 主力资金 | {arrow} {main_val:+.2f} | {main_desc} |")

        # 行业板块资金流向排名
        if overview.sector_capital_flow_top or overview.sector_capital_flow_bottom:
            lines.append("")
            if lang == "en":
                lines.append("| Sector Capital Flow Top/Bottom | Net Inflow (100M) |")
                lines.append("|----------------------------|-------------------|")
            else:
                lines.append("| 行业资金流向 TOP/BOTTOM | 净流入(亿) |")
                lines.append("|------------------------|-----------|")

            for s in overview.sector_capital_flow_top[:5]:
                inflow = s.get('net_inflow', 0)
                lines.append(f"| 🟢 {s.get('name', '-')} | +{inflow:.2f} |")
            for s in overview.sector_capital_flow_bottom[:3]:
                outflow = s.get('net_inflow', 0)
                lines.append(f"| 🔴 {s.get('name', '-')} | {outflow:.2f} |")

        return "\n".join(lines)

    def _build_market_breadth_block(self, overview: MarketOverview) -> str:
        """构建市场宽度表格（涨幅分布 + 涨停池明细）"""
        has_distribution = bool(overview.gain_distribution)
        has_limit_detail = bool(overview.limit_up_detail)
        has_yest_chg = overview.yest_limit_avg_chg != 0.0

        if not (has_distribution or has_limit_detail or has_yest_chg):
            return ""

        lang = self._get_review_language()
        lines = []

        if lang == "en":
            lines.append("#### Market Breadth & Limit-up Details")
        else:
            lines.append("#### 市场宽度与涨停明细")

        # 涨幅分布
        dist = overview.gain_distribution
        if dist:
            total = dist.get('_total', 0)
            # 移除内部字段
            display_dist = {k: v for k, v in dist.items() if not k.startswith('_')}
            if lang == "en":
                lines.append(f"| Gain Range | Count | Ratio |")
                lines.append("|-----------|-------|-------|")
            else:
                lines.append(f"| 涨幅区间 | 家数 | 占比 |")
                lines.append("|----------|------|------|")

            for range_label, count in display_dist.items():
                ratio = f"{count / total * 100:.1f}%" if total else "N/A"
                lines.append(f"| {range_label} | {count} | {ratio} |")

        # 涨停池明细
        detail = overview.limit_up_detail
        if detail:
            lines.append("")
            if lang == "en":
                lines.append(
                    f"| Limit-up | Total | First-limit | Non-first | Broken | Broken Rate |"
                )
                lines.append(
                    f"|----------|-------|-------------|----------|--------|------------|"
                )
                lines.append(
                    f"| Count | **{detail.get('total', 0)}** | "
                    f"{detail.get('first_limit', 0)} | "
                    f"{detail.get('non_first_limit', 0)} | "
                    f"{detail.get('broken_limit', 0)} | "
                    f"{detail.get('broken_rate', 0)}% |"
                )
            else:
                lines.append(
                    "| 涨停统计 | 总计 | 一字板 | 非一字板 | 炸板数 | 炸板率 |"
                )
                lines.append(
                    "|----------|------|--------|----------|--------|--------|"
                )
                lines.append(
                    f"| 数量 | **{detail.get('total', 0)}**家 | "
                    f"{detail.get('first_limit', 0)}家 | "
                    f"{detail.get('non_first_limit', 0)}家 | "
                    f"{detail.get('broken_limit', 0)}家 | "
                    f"{detail.get('broken_rate', 0)}% |"
                )

        # 昨日涨停今日表现
        yest_chg = overview.yest_limit_avg_chg
        if has_yest_chg:
            lines.append("")
            if lang == "en":
                arrow = "🟢" if yest_chg > 0 else "🔴" if yest_chg < 0 else "⚪"
                sentiment = "strong" if yest_chg > 2 else ("weak" if yest_chg < -2 else "mixed")
                lines.append(
                    f"> Yest. limit-up avg today: **{arrow} {yest_chg:+.2f}%** ({sentiment} sentiment)"
                )
            else:
                arrow = "🟢" if yest_chg > 0 else "🔴" if yest_chg < 0 else "⚪"
                sentiment = "赚钱效应强" if yest_chg > 2 else ("亏钱效应" if yest_chg < -2 else "一般")
                lines.append(
                    f"> 昨日涨停股今日均涨幅：**{arrow} {yest_chg:+.2f}%**（{sentiment}）"
                )

        return "\n".join(lines)

    def _build_index_technicals_block(self, overview: MarketOverview) -> str:
        """构建指数技术面分析表格"""
        if not overview.index_technicals:
            return ""

        lang = self._get_review_language()
        lines = []

        if lang == "en":
            lines.append("#### Index Technical Analysis")
            lines.append("| Index | MA Status | MACD | RSI | Volume | Score |")
            lines.append("|-------|----------|------|-----|--------|-------|")
        else:
            lines.append("#### 指数技术面")
            lines.append("| 指数 | 均线排列 | MACD | RSI | 量能 | 评分 |")
            lines.append("|------|----------|------|-----|------|------|")

        # 构建指数代码到名称的映射
        index_name_map = {}
        for idx in overview.indices:
            if idx.code:
                index_name_map[idx.code] = idx.name

        for code, tech in overview.index_technicals.items():
            name = index_name_map.get(code, code)
            ma_status = tech.get('ma_status', '-')
            macd_status = tech.get('macd_status', '-')
            rsi_val = tech.get('rsi', 0)
            rsi_status = tech.get('rsi_status', '-')
            vol_status = tech.get('volume_status', '-')
            score = tech.get('score', 0)

            # RSI 显示值+状态
            rsi_str = f"{rsi_val:.0f}({rsi_status})"

            # 评分颜色标记
            if score >= 70:
                score_str = f"**{score}** 🔥"
            elif score >= 50:
                score_str = f"{score}"
            else:
                score_str = f"{score} ❄️"

            lines.append(
                f"| {name} | {ma_status} | {macd_status} | {rsi_str} | {vol_status} | {score_str} |"
            )

        return "\n".join(lines)

    def _build_sector_persistence_block(self, overview: MarketOverview) -> str:
        """构建板块持续性分析表格"""
        if (not overview.persistent_leaders and not overview.new_leaders
                and not overview.falling_leaders):
            return ""

        lang = self._get_review_language()
        lines = []

        if lang == "en":
            lines.append("#### Sector Persistence Analysis")
        else:
            lines.append("#### 板块持续性分析")

        persistent = overview.persistent_leaders
        new_l = overview.new_leaders
        falling = overview.falling_leaders

        if lang == "en":
            if persistent:
                lines.append(f"- **Persistent leaders** (consecutive up): {', '.join(persistent)}")
            if new_l:
                lines.append(f"- **New leaders** (emerged today): {', '.join(new_l)}")
            if falling:
                lines.append(f"- **Falling leaders** (turned down): {', '.join(falling)}")
        else:
            if persistent:
                lines.append(f"- **连续领涨**（昨日&今日均在榜）：{'、'.join(persistent)}")
            if new_l:
                lines.append(f"- **新晋领涨**（今日首次上榜）：{'、'.join(new_l)}")
            if falling:
                lines.append(f"- **由涨转跌**（昨日领涨今日跌出）：{'、'.join(falling)}")

        # 给出判断结论
        if persistent and len(persistent) >= 2:
            if lang == "en":
                lines.append("")
                lines.append("> ✅ Strong sector continuity — leading theme likely persists.")
            else:
                lines.append("")
                lines.append("> ✅ 板块连续性强，主线题材大概率延续。")
        elif new_l and len(new_l) >= 3:
            if lang == "en":
                lines.append("")
                lines.append("> ⚠️ Rapid sector rotation — watch for sustainability.")
            else:
                lines.append("")
                lines.append("> ⚠️ 板块轮动加快，关注持续性。")
        elif falling and len(falling) >= 2:
            if lang == "en":
                lines.append("")
                lines.append("> 🔻 Multiple former leaders fading — caution on lagging sectors.")
            else:
                lines.append("")
                lines.append("> 🔻 多个前领涨板块走弱，回避滞涨方向。")

        return "\n".join(lines)

    # ── 增强数据 Prompt 文本构建（注入给 LLM 的纯文本） ─────

    def _build_capital_flow_prompt_text(self, overview: MarketOverview) -> str:
        """构建资金流向的 Prompt 文本"""
        if not overview.capital_flow and not overview.sector_capital_flow_top:
            return ""
        lines = ["## 资金流向数据"]
        cf = overview.capital_flow
        if cf:
            lines.append(f"- 北向资金净流入: {cf.get('north_net_inflow', 'N/A')} 亿元")
            if cf.get('southbound_net_inflow') is not None:
                lines.append(f"- 南向资金净流入: {cf.get('southbound_net_inflow', 'N/A')} 亿港元")
            if cf.get('main_net_inflow') is not None:
                lines.append(
                    f"- 主力资金净流入: {cf.get('main_net_inflow', 'N/A')} 亿元 "
                    f"({cf.get('main_inflow_desc', '')})"
                )
        if overview.sector_capital_flow_top:
            lines.append("- 行业资金净流入 TOP5:")
            for s in overview.sector_capital_flow_top[:5]:
                lines.append(f"  - {s['name']}: +{s.get('net_inflow', 0):.2f} 亿")
        return "\n".join(lines)

    def _build_market_breadth_prompt_text(self, overview: MarketOverview) -> str:
        """构建市场宽度的 Prompt 文本"""
        has_dist = bool(overview.gain_distribution)
        has_detail = bool(overview.limit_up_detail)
        has_yest = overview.yest_limit_avg_chg != 0.0
        if not (has_dist or has_detail or has_yest):
            return ""
        lines = ["## 市场宽度与涨停明细"]
        dist = overview.gain_distribution
        if dist:
            total = dist.get('_total', 0)
            display_dist = {k: v for k, v in dist.items() if not k.startswith('_')}
            parts = [f"{k}: {v}" for k, v in display_dist.items()]
            lines.append(f"- 涨幅分布（共{total}只）: {' | '.join(parts)}")
        detail = overview.limit_up_detail
        if detail:
            lines.append(
                f"- 涨停池: 总计{detail.get('total', 0)}家, "
                f"一字板{detail.get('first_limit', 0)}, "
                f"炸板{detail.get('broken_limit', 0)}({detail.get('broken_rate', 0)}%)"
            )
        yest_chg = overview.yest_limit_avg_chg
        if has_yest:
            sentiment = "赚钱效应强" if yest_chg > 2 else ("亏钱效应" if yest_chg < -2 else "一般")
            lines.append(f"- 昨日涨停股今日均涨幅: {yest_chg:+.2f}% ({sentiment})")
        return "\n".join(lines)

    def _build_index_technicals_prompt_text(self, overview: MarketOverview) -> str:
        """构建指数技术面的 Prompt 文本"""
        if not overview.index_technicals:
            return ""
        index_name_map = {idx.code: idx.name for idx in overview.indices if idx.code}
        lines = ["## 指数技术面分析"]
        for code, tech in overview.index_technicals.items():
            name = index_name_map.get(code, code)
            lines.append(
                f"- **{name}**: MA={tech.get('ma_status', '-')}, "
                f"MACD={tech.get('macd_status', '-')}, "
                f"RSI={tech.get('rsi', 0):.0f}({tech.get('rsi_status', '-')}), "
                f"量能={tech.get('volume_status', '-')}, "
                f"综合评分={tech.get('score', 0)}"
            )
        return "\n".join(lines)

    def _build_sector_persistence_prompt_text(self, overview: MarketOverview) -> str:
        """构建板块持续性的 Prompt 文本"""
        if (not overview.persistent_leaders and not overview.new_leaders
                and not overview.falling_leaders):
            return ""
        lines = ["## 板块持续性分析"]
        p = overview.persistent_leaders
        n = overview.new_leaders
        f = overview.falling_leaders
        if p:
            lines.append(f"- 连续领涨（昨日&今日均在榜）: {'、'.join(p)}")
        if n:
            lines.append(f"- 新晋领涨（今日首次上榜）: {'、'.join(n)}")
        if f:
            lines.append(f"- 由涨转跌（昨日领涨今日跌出）: {'、'.join(f)}")
        # 结论
        if len(p) >= 2:
            lines.append("> 判断：板块连续性强，主线题材大概率延续。")
        elif len(n) >= 3:
            lines.append("> 判断：板块轮动加快，关注持续性。")
        elif len(f) >= 2:
            lines.append("> 判断：多个前领涨板块走弱，回避滞涨方向。")
        return "\n".join(lines)

    def _build_smart_picks_prompt_text(self, overview: MarketOverview) -> str:
        """构建智能选股的 Prompt 文本（注入给 LLM 的结构化数据）"""
        if not overview.recommended_stocks:
            return ""

        source_labels = {
            'limit_up': '涨停池强势',
            'sector_leader': '连续领涨龙头',
            'hot_sector': '新晋热点龙头',
            'capital_inflow': '资金流入龙头',
        }

        lines = ["## 🎯 建议关注个股 TOP 7（AI 智能选股）"]
        lines.append("以下股票通过多维度策略综合筛选：")
        lines.append("1. **涨停池强势股**（非一字板，资金认可度高）")
        lines.append("2. **连续领涨板块龙头**（板块持续性强，龙头确定性高）")
        lines.append("3. **新晋热点板块放量龙头**（新热点启动信号）")
        lines.append("4. **资金大幅流入板块领头股**（主力资金方向）")
        lines.append("")
        lines.append("| # | 代码 | 名称 | 涨跌幅 | 板块 | 来源 | 推荐理由 | 信心度 |")
        lines.append("|---|------|------|--------|------|------|---------|--------|")

        for rank, stock in enumerate(overview.recommended_stocks[:7], 1):
            src_label = source_labels.get(stock.pick_source, '推荐')
            lines.append(
                f"| {rank} | {stock.code} | {stock.name} | "
                f"{stock.change_pct:+.1f}% | {stock.sector or '-'} | "
                f"{src_label} | {stock.reason} | {stock.confidence:.0f}/100 |"
            )

        lines.append("")
        lines.append(
            "> ⚠️ 以上选股结果基于技术面+资金面+板块联动量化筛选，"
            "仅供参考，不构成投资建议。请结合个人风险偏好独立判断。"
        )
        return "\n".join(lines)

    @staticmethod
    def _format_optional_number(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}"

    @staticmethod
    def _format_optional_pct(value: float) -> str:
        return "N/A" if value in (None, 0, 0.0) else f"{value:.2f}%"

    @staticmethod
    def _format_signed_pct(value: Any) -> str:
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{numeric_value:+.2f}%"

    @staticmethod
    def _escape_table_cell(value: str) -> str:
        return value.replace("|", "\\|")

    @staticmethod
    def _build_temperature_bar(score: int) -> str:
        filled = max(0, min(10, round(score / 10)))
        return "█" * filled + "░" * (10 - filled)

    @staticmethod
    def _describe_turnover(total_amount: float) -> str:
        if total_amount >= 15000:
            return "高活跃度"
        if total_amount >= 9000:
            return "中等活跃"
        if total_amount > 0:
            return "缩量观望"
        return "暂无数据"

    def _build_market_temperature(self, overview: MarketOverview) -> tuple[int, str]:
        participants = overview.up_count + overview.down_count
        breadth_score = 50
        if participants:
            breadth_score = int(overview.up_count / participants * 100)

        index_changes = [idx.change_pct for idx in overview.indices if idx.change_pct is not None]
        index_score = 50
        if index_changes:
            avg_change = sum(index_changes) / len(index_changes)
            index_score = int(max(0, min(100, 50 + avg_change * 12)))

        limit_total = overview.limit_up_count + overview.limit_down_count
        limit_score = 50
        if limit_total:
            limit_score = int(overview.limit_up_count / limit_total * 100)

        score = int(round(breadth_score * 0.45 + index_score * 0.35 + limit_score * 0.20))
        if self._get_review_language() == "en":
            if score >= 70:
                label = "risk-on"
            elif score >= 55:
                label = "constructive"
            elif score >= 40:
                label = "mixed"
            else:
                label = "defensive"
        else:
            if score >= 70:
                label = "强势"
            elif score >= 55:
                label = "偏暖"
            elif score >= 40:
                label = "震荡"
            else:
                label = "偏弱"
        return score, label

    def _build_review_prompt(self, overview: MarketOverview, news: List) -> str:
        """构建复盘报告 Prompt"""
        review_language = self._get_review_language()

        # 指数行情信息（简洁格式，不用emoji）
        indices_text = ""
        for idx in overview.indices:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- {idx.name}: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        top_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.top_sectors[:3]])
        bottom_sectors_text = ", ".join([f"{s['name']}({s['change_pct']:+.2f}%)" for s in overview.bottom_sectors[:3]])
        
        # 推荐个股
        stocks_text = ""
        for stock in overview.recommended_stocks[:5]:
            stocks_text += f"- {stock.code} {stock.name}: {stock.reason}\n"
        
        # 新闻信息 - 支持 SearchResult 对象或字典
        news_text = ""
        for i, n in enumerate(news[:6], 1):
            # 兼容 SearchResult 对象和字典
            if hasattr(n, 'title'):
                title = n.title[:50] if n.title else ''
                snippet = n.snippet[:100] if n.snippet else ''
            else:
                title = n.get('title', '')[:50]
                snippet = n.get('snippet', '')[:100]
            news_text += f"{i}. {title}\n   {snippet}\n"

        # ── 增强数据：构建 Prompt 文本 ──
        capital_flow_text = self._build_capital_flow_prompt_text(overview)
        market_breadth_text = self._build_market_breadth_prompt_text(overview)
        index_tech_text = self._build_index_technicals_prompt_text(overview)
        sector_persist_text = self._build_sector_persistence_prompt_text(overview)
        smart_picks_text = self._build_smart_picks_prompt_text(overview)

        # 按 region 组装市场概况与板块区块（美股无涨跌家数、板块数据）
        stats_block = ""
        sector_block = ""
        stocks_block = ""
        if review_language == "en":
            if self.profile.has_market_stats:
                stats_block = f"""## Market Breadth
- Advancers: {overview.up_count} | Decliners: {overview.down_count} | Flat: {overview.flat_count}
- Limit-up: {overview.limit_up_count} | Limit-down: {overview.limit_down_count}
- Turnover: {overview.total_amount:.0f} ({self._get_turnover_unit_label()})"""
            else:
                stats_block = "## Market Breadth\n(No equivalent advance/decline statistics are available for this market.)"

            if self.profile.has_sector_rankings:
                sector_block = f"""## Sector Performance
Leading: {top_sectors_text if top_sectors_text else "N/A"}
Lagging: {bottom_sectors_text if bottom_sectors_text else "N/A"}"""
            else:
                sector_block = "## Sector Performance\n(Sector data not available for this market.)"
            
            if stocks_text:
                stocks_block = f"""## Recommended Stocks
{stocks_text}"""
        else:
            if self.profile.has_market_stats:
                stats_block = f"""## 市场概况
- 上涨: {overview.up_count} 家 | 下跌: {overview.down_count} 家 | 平盘: {overview.flat_count} 家
- 涨停: {overview.limit_up_count} 家 | 跌停: {overview.limit_down_count} 家
- 两市成交额: {overview.total_amount:.0f} 亿元"""
            else:
                stats_block = "## 市场概况\n（该市场暂无涨跌家数等统计）"

            if self.profile.has_sector_rankings:
                sector_block = f"""## 板块表现
领涨: {top_sectors_text if top_sectors_text else "暂无数据"}
领跌: {bottom_sectors_text if bottom_sectors_text else "暂无数据"}"""
            else:
                sector_block = "## 板块表现\n（该市场暂无板块涨跌数据）"
            
            if stocks_text:
                stocks_block = f"""## 值得关注的个股
{stocks_text}"""

        data_no_indices_hint = (
            "注意：由于行情数据获取失败，请主要根据【市场新闻】进行定性分析和总结，不要编造具体的指数点位。"
            if not indices_text
            else ""
        )
        if review_language == "en":
            data_no_indices_hint = (
                "Note: Market data fetch failed. Rely mainly on [Market News] for qualitative analysis. Do not invent index levels."
                if not indices_text
                else ""
            )
            indices_placeholder = indices_text if indices_text else "No index data (API error)"
            news_placeholder = news_text if news_text else "No relevant news"
        else:
            indices_placeholder = indices_text if indices_text else "暂无指数数据（接口异常）"
            news_placeholder = news_text if news_text else "暂无相关新闻"

        if review_language == "en":
            report_title = self._get_review_title(overview.date).removeprefix("## ").strip()
            return f"""You are a professional US/A/H market analyst. Please produce a concise market recap report based on the data below.

[Requirements]
- Output pure Markdown only
- No JSON
- No code blocks
- Use emoji sparingly in headings (at most one per heading)
- The entire fixed shell, headings, guidance, and conclusion must be in English

---

# Today's Market Data

## Date
{overview.date}

## Major Indices
{indices_placeholder}

{stats_block}

{sector_block}

{stocks_block}

## Market News
{news_placeholder}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# Output Template (follow this structure)

## {report_title}

### 1. Market Summary
(2-3 sentences summarizing overall market tone, index moves, and liquidity.)

### 2. Index Commentary
({self._get_index_hint()})

### 3. Fund Flows
(Interpret what turnover, participation, and flow signals imply.)

### 4. Sector Highlights
(Analyze the drivers behind the leading and lagging sectors or themes.)

### 5. Stock Recommendations
(Focus on the 5 recommended stocks and their investment rationale.)

### 6. Outlook
(Provide the near-term outlook based on price action and news.)

### 7. Risk Alerts
(List the main risks to monitor.)

### 8. Strategy Plan
(Provide an offensive/balanced/defensive stance, a position-sizing guideline, one invalidation trigger, and end with "For reference only, not investment advice.")

---

Output the report content directly, no extra commentary.
"""

        # A 股场景使用中文提示语
        return f"""你是一位专业的A/H/美股市场分析师，请根据以下数据生成一份结构化的{self._get_market_scope_name('zh')}大盘复盘报告。

【重要】输出要求：
- 必须输出纯 Markdown 文本格式
- 禁止输出 JSON 格式
- 禁止输出代码块
- emoji 仅在标题处少量使用（每个标题最多1个）
- 报告要像交易员盘后工作台：先给结论，再按数据表、主线、催化、计划展开
- 不要重复列出已由系统注入的表格数据；正文负责解释表格背后的含义

---

# 今日市场数据

## 日期
{overview.date}

## 主要指数
{indices_placeholder}

{stats_block}

{sector_block}

{stocks_block}

## 市场新闻
{news_placeholder}

{capital_flow_text if capital_flow_text else ""}

{market_breadth_text if market_breadth_text else ""}

{index_tech_text if index_tech_text else ""}

{sector_persist_text if sector_persist_text else ""}

{smart_picks_text if smart_picks_text else ""}

{data_no_indices_hint}

{self._get_strategy_prompt_block()}

---

# 输出格式模板（请严格按此格式输出）

## {overview.date} 大盘复盘

> 一句话给出今日市场状态、核心矛盾和明日优先观察方向。

### 一、盘面总览
（2-3句话概括指数、涨跌家数、成交额和情绪温度，明确"强势/偏暖/震荡/偏弱"判断）

### 二、指数结构
（{self._get_index_hint()}，说明谁在护盘、谁在拖累，以及关键支撑/压力）

### 三、板块主线
（分析领涨/领跌板块背后的逻辑、持续性和是否形成主线；分析板块资金量权重）

### 四、值得关注的个股 TOP 5
（基于领涨板块，选出5支龙头股或高成长潜力个股，给出简要投资逻辑）

### 五、资金与情绪
（解读成交额、涨跌停结构、市场宽度和风险偏好）

### 六、消息催化
（结合近三日新闻，提炼真正影响明日交易的催化或扰动）

### 七、明日交易计划
（给出进攻/均衡/防守结论、仓位区间、关注方向、回避方向和一个触发失效条件）

### 八、风险提示
（列出需要关注的风险点；最后补充"建议仅供参考，不构成投资建议"。）

---

请直接输出复盘报告内容，不要输出其他说明文字。
"""
    
    def _generate_template_review(self, overview: MarketOverview, news: List) -> str:
        """使用模板生成复盘报告（无大模型时的备选方案）"""
        template_language = self._get_template_review_language()
        mood_code = self.profile.mood_index_code
        # 根据 mood_index_code 查找对应指数
        # cn: mood_code="000001"，idx.code 可能为 "sh000001"（以 mood_code 结尾）
        # us: mood_code="SPX"，idx.code 直接为 "SPX"
        mood_index = next(
            (
                idx
                for idx in overview.indices
                if idx.code == mood_code or idx.code.endswith(mood_code)
            ),
            None,
        )
        if mood_index:
            if mood_index.change_pct > 1:
                market_mood = self._get_market_mood_text("strong_up", template_language)
            elif mood_index.change_pct > 0:
                market_mood = self._get_market_mood_text("mild_up", template_language)
            elif mood_index.change_pct > -1:
                market_mood = self._get_market_mood_text("mild_down", template_language)
            else:
                market_mood = self._get_market_mood_text("strong_down", template_language)
        else:
            market_mood = self._get_market_mood_text("range", template_language)
        
        # 指数行情（简洁格式）
        indices_text = ""
        for idx in overview.indices[:4]:
            direction = "↑" if idx.change_pct > 0 else "↓" if idx.change_pct < 0 else "-"
            indices_text += f"- **{idx.name}**: {idx.current:.2f} ({direction}{abs(idx.change_pct):.2f}%)\n"
        
        # 板块信息
        separator = ", " if template_language == "en" else "、"
        top_text = separator.join([s['name'] for s in overview.top_sectors[:3]])
        bottom_text = separator.join([s['name'] for s in overview.bottom_sectors[:3]])

        if template_language == "en":
            stats_section = ""
            if self.profile.has_market_stats:
                stats_section = f"""
### 3. Breadth & Liquidity
| Metric | Value |
|--------|-------|
| Advancers | {overview.up_count} |
| Decliners | {overview.down_count} |
| Limit-up | {overview.limit_up_count} |
| Limit-down | {overview.limit_down_count} |
| Turnover ({self._get_turnover_unit_label()}) | {overview.total_amount:.0f} |
"""
            sector_section = ""
            if self.profile.has_sector_rankings and (top_text or bottom_text):
                sector_section = f"""
### 4. Sector Highlights
- **Leaders**: {top_text or "N/A"}
- **Laggards**: {bottom_text or "N/A"}
"""
            market_names = {"us": "US Market Recap", "hk": "HK Market Recap"}
            market_name = market_names.get(self.region, "A-share Market Recap")
            report = f"""## {overview.date} {market_name}

### 1. Market Summary
Today's {self._get_market_scope_name(template_language)} showed **{market_mood}**.

### 2. Major Indices
{indices_text or "- No index data available"}
{stats_section}
{sector_section}
### 5. Risk Alerts
Market conditions can change quickly. The data above is for reference only and does not constitute investment advice.

{self._get_strategy_markdown_block(template_language)}

---
*Review Time: {datetime.now().strftime('%H:%M')}*
"""
            return report

        market_labels = {"cn": "A股", "us": "美股", "hk": "港股"}
        market_label = market_labels.get(self.region, "A股")
        dashboard_block = self._build_stats_block(overview)
        indices_block = self._build_indices_block(overview)
        sector_block = self._build_sector_block(overview)
        sector_capital_block = self._build_sector_capital_block(overview)
        stock_recommendation_block = self._build_stock_recommendation_block(overview)
        return f"""## {overview.date} 大盘复盘

> 今日{market_label}市场整体呈现**{market_mood}**态势，优先观察指数承接、成交额变化和板块持续性。

### 一、盘面总览
{dashboard_block or "暂无市场宽度数据。"}

### 二、指数结构
{indices_block or indices_text or "暂无指数数据。"}

### 三、板块主线
{sector_capital_block or sector_block or "- 暂无板块涨跌榜数据。"}

### 四、值得关注的个股 TOP 5
{stock_recommendation_block or "暂无个股推荐。"}

### 五、资金与情绪
- 结合成交额和涨跌家数看，当前更适合等待确认，避免仅凭单一热点追高。

### 六、消息催化
- 暂无可用新闻时，应降低对题材持续性的确定性判断。

### 七、明日交易计划
- **结论**：均衡观察。
- **仓位**：控制在中性区间，等待指数与主线共振。
- **关注方向**：{top_text or "强于指数的主线板块"}。
- **回避方向**：{bottom_text or "连续走弱且缺少修复信号的方向"}。

### 八、风险提示
- 市场有风险，投资需谨慎。以上数据仅供参考，不构成投资建议。

---
*复盘时间: {datetime.now().strftime('%H:%M')}*
"""
    
    def run_daily_review(self) -> str:
        """
        执行每日大盘复盘流程
        
        Returns:
            复盘报告文本
        """
        logger.info("========== 开始大盘复盘分析 ==========")
        
        # 1. 获取市场概览
        overview = self.get_market_overview()
        
        # 2. 搜索市场新闻
        news = self.search_market_news()
        
        # 3. 生成复盘报告
        report = self.generate_market_review(overview, news)
        
        logger.info("========== 大盘复盘分析完成 ==========")
        
        return report


# 测试入口
if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    )
    
    analyzer = MarketAnalyzer()
    
    # 测试获取市场概览
    overview = analyzer.get_market_overview()
    print(f"\n=== 市场概览 ===")
    print(f"日期: {overview.date}")
    print(f"指数数量: {len(overview.indices)}")
    for idx in overview.indices:
        print(f"  {idx.name}: {idx.current:.2f} ({idx.change_pct:+.2f}%)")
    print(f"上涨: {overview.up_count} | 下跌: {overview.down_count}")
    print(f"成交额: {overview.total_amount:.0f}亿")
    
    # 测试生成模板报告
    report = analyzer._generate_template_review(overview, [])
    print(f"\n=== 复盘报告 ===")
    print(report)
