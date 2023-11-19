import random
random.seed(0)

# Đọc tập tin gốc
with open('/mlcv/WorkingSpace/Personals/datnt/datasets/labels_have_empty.txt', 'r') as file:
    lines = file.readlines()

# Loại bỏ khoảng trắng và ký tự xuống dòng khỏi mỗi dòng
lines = [line.rstrip('\n') for line in lines]
random.shuffle(lines)

# Xác định số lượng dòng muốn chọn ngẫu nhiên
num_lines_to_select = int(len(lines)*0.1)

# Kiểm tra xem số lượng dòng cần chọn có vượt quá số dòng trong tập tin không
if num_lines_to_select > len(lines):
    print("Số lượng dòng cần chọn lớn hơn số dòng trong tập tin.")
else:
    # Chọn ngẫu nhiên 10% dòng không trùng nhau
    random_lines = random.sample(lines, num_lines_to_select)

    # Lưu các dòng ngẫu nhiên vào tập valid
    with open('/mlcv/WorkingSpace/Personals/datnt/datasets/labels_val_have_empty.txt', 'w') as output_file:
        for line in random_lines:
            output_file.write(line + '\n')
    
    # Lưu các dòng còn lại vào tập train mới
    remaining_lines = [line for line in lines if line not in random_lines]
    with open('/mlcv/WorkingSpace/Personals/datnt/datasets/labels_train_have_empty.txt', 'w') as remaining_file:
        for line in remaining_lines:
            remaining_file.write(line + '\n')

    print("Đã chia dữ liệu")
