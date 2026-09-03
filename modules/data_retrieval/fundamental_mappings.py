"""Define canonical metrics and constrained XBRL mappings for fundamentals retrieval."""

from typing import Literal, TypeAlias

Metric: TypeAlias = Literal[
    'revenue', 'cost_of_goods_sold', 'gross_profit', 'operating_income', 'net_income',
    'total_assets', 'total_liabilities', 'shareholders_equity', 'cash_and_equivalents',
    'current_assets', 'current_liabilities', 'property_plant_equipment', 'long_term_debt',
    'short_term_debt', 'operating_cash_flow', 'capital_expenditures',
    'depreciation_amortization', 'stock_issuance', 'stock_repurchases',
]

METRICS: dict[Metric, tuple[str, tuple[str, ...]]] = {
    'revenue': ('income_statement', ('Revenue', 'Revenues', 'SalesRevenueNet', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'ElectricUtilityRevenue')),
    'cost_of_goods_sold': ('income_statement', ('CostOfRevenue', 'CostOfGoodsAndServicesSold', 'CostOfGoodsSold', 'CostOfGoodsSoldElectric')),
    'gross_profit': ('income_statement', ('GrossProfit',)),
    'operating_income': ('income_statement', ('OperatingIncomeLoss',)),
    'net_income': ('income_statement', ('NetIncomeLoss', 'ProfitLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic')),
    'total_assets': ('balance_sheet', ('Assets',)),
    'total_liabilities': ('balance_sheet', ('Liabilities',)),
    'shareholders_equity': ('balance_sheet', ('StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest')),
    'cash_and_equivalents': ('balance_sheet', ('CashAndCashEquivalentsAtCarryingValue', 'CashEquivalentsAtCarryingValue', 'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents')),
    'current_assets': ('balance_sheet', ('AssetsCurrent',)),
    'current_liabilities': ('balance_sheet', ('LiabilitiesCurrent',)),
    'property_plant_equipment': ('balance_sheet', ('PropertyPlantAndEquipmentNet',)),
    'long_term_debt': ('balance_sheet', ('LongTermDebtNoncurrent', 'LongTermDebtAndCapitalLeaseObligations', 'DebtAndCapitalLeaseObligations', 'LongTermDebt')),
    'short_term_debt': ('balance_sheet', ('DebtCurrent', 'ShortTermDebtCurrent')),
    'operating_cash_flow': ('cash_flow_statement', ('NetCashProvidedByUsedInOperatingActivities', 'NetCashProvidedByUsedInOperatingActivitiesContinuingOperations')),
    'capital_expenditures': ('cash_flow_statement', ('PaymentsToAcquirePropertyPlantAndEquipment', 'PaymentsToAcquireProductiveAssets', 'PaymentsForProceedsFromOtherPropertyPlantAndEquipment')),
    'depreciation_amortization': ('cash_flow_statement', ('DepreciationDepletionAndAmortization', 'DepreciationAndAmortization', 'DepreciationDepletionAndAmortizationPropertyPlantAndEquipment')),
    'stock_issuance': ('cash_flow_statement', ('ProceedsFromStockOptionsExercised', 'ProceedsFromIssuanceOfCommonStock', 'ProceedsFromSaleOfTreasuryStock', 'ProceedsFromStockPlans', 'ProceedsFromIssuanceOfSharesUnderIncentiveAndShareBasedCompensationPlansIncludingStockOptions')),
    'stock_repurchases': ('cash_flow_statement', ('PaymentsForRepurchaseOfCommonStock',)),
}

META_COLUMNS = {'concept', 'label', 'level', 'abstract', 'unit', 'balance', 'weight', 'preferred_sign', 'standard_concept', 'point_in_time', 'dimension'}
METRIC_DEPENDENCIES: dict[Metric, tuple[Metric, ...]] = {
    'gross_profit': ('revenue', 'cost_of_goods_sold'),
    'total_liabilities': ('total_assets', 'shareholders_equity'),
}
STANDARD_CONCEPTS: dict[Metric, tuple[str, ...]] = {
    'revenue': ('ElectricUtilityRevenue', 'Revenue'),
    'cost_of_goods_sold': ('CostOfGoodsAndServicesSold',),
    'operating_income': ('OperatingIncomeLoss',),
    'net_income': ('NetIncomeLoss', 'ProfitLoss'),
    'cash_and_equivalents': ('CashAndMarketableSecurities',),
    'property_plant_equipment': ('PlantPropertyEquipmentNet',),
    'operating_cash_flow': ('NetCashFromOperatingActivities', 'OperatingCashFlow'),
    'capital_expenditures': ('CapitalExpenses',),
    'stock_issuance': ('StockIssuanceProceeds',),
    'stock_repurchases': ('StockRepurchasePayments', 'CommonStockRepurchasePayments'),
    'long_term_debt': ('LongTermDebt',),
}
COMPONENT_STANDARD_CONCEPTS: dict[Metric, tuple[str, ...]] = {
    'shareholders_equity': (
        'CommonEquity',
        'AdditionalPaidInCapital',
        'RetainedEarnings',
        'AccumulatedOtherComprehensiveIncome',
        'PreferredStock',
        'TreasuryStock',
        'NonControllingInterest',
    ),
    'depreciation_amortization': ('DepreciationExpense', 'AmortizationOfIntangibles'),
    'short_term_debt': ('ShortTermDebt', 'CurrentPortionOfLongTermDebt'),
}
LABEL_PATTERNS: dict[Metric, tuple[str, ...]] = {
    'revenue': (r'^total revenue$', r'^revenue$'),
    'cost_of_goods_sold': (r'^total cost of sales$', r'^cost of (?:goods|products|services|sales)$'),
    'operating_income': (r'^operating income(?: \(loss\))?$',),
    'property_plant_equipment': (r'^propert(?:y|ies)(?:,? plant)?(?:,? and)? equipment, net$',),
    'operating_cash_flow': (
        r'^net cash (?:provided by|used in|provided by\s*\(?used in\)?|provided by/used in) operating activities(?: continuing operations)?$',
    ),
    'depreciation_amortization': (
        r'^depreciation(?:,| and) amortization(?:,? and other)?$',
        r'^depreciation, depletion and amortization$',
        r'^depreciation and amortization expense$',
    ),
    'shareholders_equity': (
        r"^(?:total )?(?:common )?(?:shareholders|stockholders)(?:'|’)?\s+equity$",
    ),
    'stock_issuance': (
        r'^proceeds from (?:the )?(?:issuance|sale) of (?:common|treasury) stock.*$',
        r'^proceeds from (?:stock options exercised|stock plans|employee stock plans).*$',
    ),
    'long_term_debt': (r'^long-term debt(?: and (?:capital )?lease obligations)?$',),
}
LABEL_EXCLUSIONS: dict[Metric, tuple[str, ...]] = {
    'stock_issuance': ('repurchase', 'compensation expense', 'tax benefit', 'issuance cost'),
}
COMPONENT_LABEL_PATTERNS: dict[Metric, tuple[str, ...]] = {
    'depreciation_amortization': (
        r'^depreciation(?: expense)?$',
        r'^depreciation and other amortization$',
        r'^amortization (?:of )?(?:intangible assets|intangibles)$',
        r'^depreciation and impairment of equipment on operating leases, net$',
        r'^depreciation, amortization and impairment charges on property, net$',
        r'^depreciation, amortization, and decommissioning$',
        r'^depreciation, amortization, depletion and accretion$',
    ),
    'short_term_debt': (
        r'^short-term borrowings$',
        r'^current portion of long-term debt(?: and (?:capital )?lease obligations)?$',
    ),
}
COMPONENT_CONCEPT_PATTERNS: dict[Metric, tuple[str, ...]] = {
    'depreciation_amortization': (
        r'.*(?:_|:)(?:DepreciationAmortizationAndOther|MainlineAndExpressDepreciationAndAmortization)$',
    ),
    'short_term_debt': (
        r'.*(?:_|:)RecourseDebtCurrent$',
        r'.*(?:_|:)NonRecourseDebtCurrent(?:Balance)?$',
        r'.*(?:_|:)NonRecourseBorrowingsOfConsolidatedSecuritizationEntitiesCurrent$',
        r'.*(?:_|:)LongTermDebtAndOtherDebtNetCurrent$',
    ),
    'long_term_debt': (
        r'.*(?:_|:)RecourseDebtNonCurrent$',
        r'.*(?:_|:)NonRecourseDebtNonCurrent(?:Balance)?$',
        r'.*(?:_|:)NonRecourseBorrowingsOfConsolidatedSecuritizationEntitiesNonCurrent$',
        r'.*(?:_|:)LongTermDebtAndOtherDebtNetNoncurrent$',
    ),
}

__all__ = [
    'COMPONENT_CONCEPT_PATTERNS',
    'COMPONENT_LABEL_PATTERNS',
    'COMPONENT_STANDARD_CONCEPTS',
    'LABEL_EXCLUSIONS',
    'LABEL_PATTERNS',
    'META_COLUMNS',
    'METRIC_DEPENDENCIES',
    'METRICS',
    'Metric',
    'STANDARD_CONCEPTS',
]
