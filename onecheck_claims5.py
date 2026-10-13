cms_opd = """15820 15821 15822 15823 67900 67901 67902 67903 67904 67906 67908
64612 64615 J0585 J0586 J0587 J0588
15830 15847 15877
20912 21210 30400 30410 30420 30430 30435 30450 30460 30462 30465 30520
36473 36474 36475 36476 36478 36479 36482 36483
22551 22552 63650
64490 64491 64493 64494 64633 64634 64635 64636""".split()
cms = set(exploded_rules_v2.loc[exploded_rules_v2.PAYOR_CODE == "CMSHSP", "cpt_code"])
found = [c for c in cms_opd if c in cms]
print(f"CMS OPD codes on CMSHSP list: {len(found)} of {len(cms_opd)}")
print("Missing:", [c for c in cms_opd if c not in cms])
https://www.cms.gov/data-research/monitoring-programs/medicare-fee-service-compliance-programs/prior-authorization-pre-claim-review-initiatives/prior-authorization-certain-hospital-outpatient-department-opd-services
https://www.cms.gov/files/document/opd-services-require-prior-authorization.pdf
