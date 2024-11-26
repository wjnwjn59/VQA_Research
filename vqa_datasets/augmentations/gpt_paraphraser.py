import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


prompt_template = """
Hãy đóng vai một chuyên gia trong việc đặt câu hỏi dựa trên câu hỏi gốc để tạo ra các câu viết lại dùng để huấn luyện mô hình VQA.  
Mục tiêu của bạn là tạo ra các câu diễn đạt lại một cách đa dạng, tự nhiên và phù hợp với ngữ cảnh nhằm tăng cường sự phong phú về ngôn ngữ cho tập dữ liệu.  

Hãy tuân theo các yêu cầu sau:  
Tạo đúng {0} câu diễn đạt lại cho câu hỏi tiếng Việt được cung cấp, đảm bảo:  
1. **Giữ nguyên ý nghĩa**: Bảo toàn ý nghĩa ban đầu mà không thay đổi mục đích của câu hỏi.  
2. **Phù hợp ngữ cảnh**: Đảm bảo các câu viết lại phù hợp với ngữ cảnh của các câu hỏi VQA thường gặp, tránh trùng lặp ý (ví dụ: miêu tả, giải thích hoặc đưa thông tin thực tế).  
3. **Đa dạng và kiểm soát**: Sử dụng từ vựng phong phú, từ đồng nghĩa, ngữ điệu, và cấu trúc câu đa dạng để tạo ra các câu khác biệt, không lặp lại.  
4. **Trôi chảy**: Đảm bảo tất cả các câu đều trôi chảy, tự nhiên và phù hợp với người dùng tiếng Việt.  
5. **Nội dung và độ dài tối đa**: Chỉ sử dụng câu đơn giản và không sử dụng dấu câu, mỗi câu viết lại phải dài dưới 20 từ.  

**Định dạng đầu ra**:  
- Chỉ cung cấp các câu diễn đạt lại và mỗi câu nằm trên một dòng riêng biệt.

### Ví dụ tạo đúng 3 câu diễn đạt lại:  
**Đầu vào**:  
"Bạn đã xem bộ phim nào thú vị gần đây chưa?"  

**Đầu ra**:  
Gần đây bạn có xem qua bộ phim nào hấp dẫn không?
Dạo này bạn đã xem bộ phim nào mà bạn thấy hay không?
Bạn có tình cờ xem bộ phim thú vị nào gần đây không?

### Lưu ý:  
- Luôn tạo đúng {0} câu diễn đạt lại.  
- Các câu viết lại được thiết kế dành cho huấn luyện mô hình VQA.  
- Nội dung câu viết lại không được chứa dấu câu hoặc ký tự bất thường.  
- Đảm bảo mỗi dòng chứa một câu viết lại duy nhất.  

### Câu hỏi đầu vào: "{1}"  
"""


client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


def req_gpt(prompt_question):
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt_question}"
                }
            ],
            temperature=0.9,        # Độ sáng tạo cao để tạo ra nhiều phiên bản
            top_p=0.9,              # Đảm bảo tính phong phú trong cách diễn đạt
            frequency_penalty=0.3,  # Giảm lặp lại từ/cụm từ
            presence_penalty=0.6    # Khuyến khích dùng từ mới
        )
        response = completion.choices[0].message.content
    except openai.OpenAIError as e:
        print(f"Call API Error: {e}")
        response = ""

    return response


def get_gpt_paraphrase(question, num_paraphrase, backup_amount=1):
    prompt = prompt_template.format(num_paraphrase + backup_amount, question)
    response = req_gpt(prompt)
    paraphrases = [line.strip()
                   for line in response.split('\n') if line.strip()]
    paraphrases = paraphrases[:num_paraphrase]
    return paraphrases
