import pandas as pd

from address_correction import AddressCorrection
address_correction = AddressCorrection()

def correct(answer):
    if not answer:
        return ""
    answer = answer.lower()
    result, diff = address_correction.address_correction(answer, correct_th=40)
    result = result.replace(",", "")
    result = " ".join([x.capitalize() for x in result.split()])
    return result.strip()

df = pd.read_csv("submit_finetune_finetune.csv")


df['answer'] = df['answer'].apply(lambda x: x if pd.notna(x) else '')
df['answer'] = df['answer'].apply(correct)
df.to_csv("submit_finetune_finetune_new.csv", index=False)


# result, diff = address_correction.address_correction('963/95/26 Hung Đàng Phường 52 Quận 6 Tp Hồ Chí Minh'.lower())
# # return ('thọ nghiệp, xuân trường, nam định', 1.6)
# # giá trị trả về đầu là địa chỉ đã sửa, giá trị trả về thứ 2 là chỉ số đo sự sai khác giữa địa chỉ đầu vào và địa chỉ trả về. Giá trị trả về là -1 nếu không thể sửa được
# result =result.replace(",", "")
# print(result.title())
# print(diff)
