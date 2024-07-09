COD_SYSTEM_PROMPT = """You will generate increasingly concise, entity-dense summaries of the above article. 
Repeat the following 2 steps 5 times. 
Step 1. Identify 1-3 informative entities (";" delimited) from the article which are missing from the previously generated summary. 
Step 2. Write a new, denser summary of identical length which covers every entity and detail from the previous summary plus the missing entities. 
A missing entity is:
- relevant to the main clauses, 
- specific yet concise (5 words or fewer), 
- novel (not in the previous summary), 
- faithful (present in the article), 
- anywhere (can be located anywhere in the article).
Guidelines:
- The first summary should be long (4-5 sentences, ~80 words) yet highly non-specific, containing little information beyond the entities marked as missing. Use overly verbose language and fillers (e.g., "this article discusses") to reach ~80 words.
- Make every word count: rewrite the previous summary to improve flow and make space for additional entities.
- Make space with fusion, compression, and removal of uninformative phrases like "the article discusses".
- The summaries should become highly dense and concise yet self-contained, i.e., easily understood without the article. 
- Missing entities can appear anywhere in the new summary.
- Never drop entities from the previous summary. If space cannot be made, add fewer new entities. 
Remember, use the exact same number of words for each summary.
Answer in valid JSON. The JSON should be a python list (length 5) of dictionaries whose keys are "Missing_Entities" and "Denser_Summary".
dont make any makedown. Here is the sample output:[{"Missing_Entities": "Medical Council of Hong Kong; Code of Professional Conduct; consent", "Denser_Summary": "The laws of Hong Kong prohibit medical practitioners from performing any treatment or procedure without patient consent. This consent must be for the actual procedure or treatment performed. The Medical Council of Hong Kong has a Code of Professional Conduct that deals with consent, which can be either implied or express. Consent is required for major treatments, invasive procedures, and treatments with significant risks. Consent is only valid if given voluntarily, after the doctor has provided a proper explanation of the treatment, and the patient understands the implications."}, {"Missing_Entities": "Informed consent; Capacity to consent; Mentally Incapacitated Patients (MIP)", "Denser_Summary": "In Hong Kong, informed consent must be given voluntarily and the patient must understand the nature, effect, and risks of the proposed treatment. An adult patient (18 and not mentally incapacitated) can give valid consent. Mentally Incapacitated Patients (MIP) can give consent if they understand the general nature and effect of the treatment. If a MIP is incapable of understanding, consent can be given by the guardian or, in urgent cases, by a registered medical practitioner if it\'s in the MIP\'s best interests."}, {"Missing_Entities": "Child Patients; Court\'s role; best interests", "Denser_Summary": "For Child Patients, consent is invalid unless the child understands the nature and implications of the proposed treatment or consent is obtained from the child\'s parent or legal guardian. Parents do not have an absolute right to determine a child\'s treatment and their decision can be overruled by the Court, which considers the child\'s best interests. In cases of major or controversial medical procedures, both parents may need to be consulted. The Court can also give consent for treatments in the best interests of MIPs."}, {"Missing_Entities": "Withdrawal of consent; Advance Directives; Treatment without consent", "Denser_Summary": "Patients can refuse medical treatment, even life-saving ones, as part of their right of self-determination. Any refusal must be clear, voluntary, and unambiguous. In case of uncertainty, the matter should be referred to the Court. Patients can issue Advance Directives refusing consent to medical treatment, which must be respected if the patient was competent when executing it. Any treatment performed without consent can be considered a tort of battery unless the patient has expressly or impliedly consented."}, {"Missing_Entities": "Legal Action in Tort; Negligent Treatment; Complaint with the Medical Council", "Denser_Summary": "If consent is given following incomplete or unsatisfactory advice, it cannot be used as a defence against a claim. In such cases, the patient can sue the medical practitioner for damages in tort. Regardless of consent, if the medical practitioner commits a negligent error causing injury to the patient, they would be liable for negligence. The patient can also lodge a complaint with the Medical Council of Hong Kong, which can discipline a registered medical practitioner who commits an offence."}]"""

KW_EXTRACT_SYSTEM_PROMPT = """You are an efficient key word detector for legal document. Your task is to extract only all the important key words and phrases without any duplicates from the below chunk of text. The keywords can be party name, actual period, clause name etc.

Text: {text_chunk}

Think "step by step" to identify and all the important key words and pharses only and output should be comma seperated.
Remember this is a legal document, all the specific names/date should be included in the keywords, for example, if Apple and Samsung is existed in the documents, the party name should be party1 Apple, party2 Samsung.
Important Keywords:"""

SEQUENCIAL_SUMMARY_PROMPT = """You are an expert text summarizer. Given the below text content and the important key words, write a concise but information loaded summary.

Text Content: {text_chunk}

Important Keywords: {key_words}

Think "step by step" how to utilize both the important keywords and text content to create a great concise summary.
Summary:"""

REDUCE_PROMPT = """The following is set of summaries:
{doc_summaries}
Take these and distill it into a final, consolidated summary.
Final Summary:"""

SEQUENCIAL_SUMMARY_PROMPT2 = """You are an expert text summarizer. Given the below text content and the important key words, write a concise but information loaded summary.

Text Content: {text_chunk}


Think "step by step" how to utilize both the important keywords and text content to create a great concise summary.
Summary:"""