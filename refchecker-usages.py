#%%
from pprint import pprint
from refchecker import LLMExtractor, LLMChecker


# Modify the following values accordingly
model_name = 'meta-llama/Meta-Llama-3-8B-Instruct'


question = """What's the longest river in the world?"""

response1 = """
The longest river in the world is the Nile River, located in northeastern Africa. 
It stretches for approximately 6,853 kilometers (4,258 miles) from its sources in Burundi, Rwanda, and Tanzania to its delta on the Mediterranean Sea in Egypt.
"""

response2 = """
The longest river in the world is the Nile River. 
It flows northward through northeastern Africa for approximately 6,650 kilometers (4,132 miles) from its most distant source in the African Great Lakes region to the Mediterranean Sea.
"""

reference = """
The Nile is a major north-flowing river in northeastern Africa. 
It flows into the Mediterranean Sea. The Nile is the longest river in Africa and has historically been considered the longest river in the world, though this has been contested by research suggesting that the Amazon River is slightly longer. 
Of the world's major rivers, the Nile is one of the smallest, as measured by annual flow in cubic metres of water. 
About 6,650 km (4,130 mi) long, its drainage basin covers eleven countries: the Democratic Republic of the Congo, Tanzania, Burundi, Rwanda, Uganda, Kenya, Ethiopia, Eritrea, South Sudan, Sudan, and Egypt. 
In particular, the Nile is the primary water source of Egypt, Sudan and South Sudan. 
Additionally, the Nile is an important economic river, supporting agriculture and fishing.
"""

#%%
# Claim extraction
extractor = LLMExtractor(
    claim_format='triplet', 
    model=model_name,
    batch_size=8
)

# each element in claims is an instance of Claim
extraction_results = extractor.extract(
    batch_responses=[response1, response2],
    max_new_tokens=1000
)
for i, res in enumerate(extraction_results):
    print(f'Claims in Response {i+1}:')
    for claim in res.claims:
        print(claim.content)
    print('----')

#%%
checker = LLMChecker(model=model_name, batch_size=8)

batch_claims = []
for res in extraction_results:
    batch_claims.append([claim.content for claim in res.claims])

batch_reference = [reference] * len(batch_claims)

checking_results = checker.check(
    batch_claims=batch_claims,
    batch_references=batch_reference,
    max_reference_segment_length=0
)

predicted_labels = []
for i, (extract_res, check_res) in enumerate(zip(extraction_results, checking_results)):
    print(f'Checking results for Response {i+1}:')
    for claim, pred_label in zip(extract_res.claims, check_res):
        print(f'{claim.content} --> {pred_label}')
        predicted_labels.append(pred_label)
    print('---')
print('Final labels:')
pprint(predicted_labels)
