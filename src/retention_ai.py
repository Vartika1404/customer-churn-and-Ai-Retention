def get_retention_strategy(risk):

    if risk >= 0.80:
        return """
High Risk Customer

1. Offer 20% Discount
2. Offer Annual Contract
3. Provide Priority Support
"""

    elif risk >= 0.50:
        return """
Medium Risk Customer

1. Offer Loyalty Rewards
2. Send Promotional Offers
"""

    else:
        return """
Low Risk Customer

1. Regular Engagement
2. Thank You Email
"""



print(get_retention_strategy(0.92))