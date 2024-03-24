import jiwer

#with open("20240130_631_correctc.txt","r",encoding="utf-8") as correctedasr:
#  reference=correctedasr.read()
#with open("20240130c.txt","r",encoding="utf-8") as openaiasr:
#  hypothesis=openaiasr.read()

gloden_file = "20221025_566_correctc.txt"
target_file = "20221025_org.txt"
#target_file = "20221025_vad.txt"

#gloden_file = "20230829_609_correctc.txt"
#target_file = "20230829_org.txt"
#target_file = "20230829_vad.txt"

#gloden_file = "20231225_626_correctc.txt"
#target_file = "20231225_org.txt"
#target_file = "20231225_vad.txt"

#gloden_file = "20240130_631_correctc.txt"
#target_file = "20240130_org.txt"
#target_file = "20240130_vad.txt"

with open(gloden_file,"r",encoding="utf-8") as correctedasr:
  reference=correctedasr.read()
with open(target_file,"r",encoding="utf-8") as openaiasr:
  hypothesis=openaiasr.read()

#reference = "今天天氣很好嗎"
#hypothesis = "今天天氣很好啊"
error = jiwer.cer(reference, hypothesis)
print(error)
output = jiwer.process_characters(reference, hypothesis)
#print(jiwer.visualize_alignment(output))

#print(jiwer.visualize_alignment(output))
