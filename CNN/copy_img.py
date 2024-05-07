import os
import shutil
NUMBER_IMG = 200
# Đường dẫn tới thư mục chứa tất cả các ảnh
source_folder = r'D:\code\introToAi\data\asl_alphabet_train\asl_alphabet_train'

# Đường dẫn tới thư mục mới mà bạn muốn tạo và copy các ảnh vào
target_folder = r'D:\code\introToAi\data\asl_alphabet_train\train_200_img_one_folder'

def copy_in_one_folder_to_many_folder(source_folder, target_folder):
    # Tạo thư mục mới nếu chưa tồn tại
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # Duyệt qua tất cả các tệp trong thư mục gốc
    for filename in os.listdir(source_folder):
        # Tách tên tệp và phần mở rộng
        name, ext = os.path.splitext(filename)
        # Kiểm tra nếu tệp là ảnh và tên chứa một trong các chữ cái từ A-Z và số thứ tự từ 1 đến 1000
        if ext.lower() == '.jpg' and name[0].isalpha() and name[1:].isdigit() and int(name[1:]) <= NUMBER_IMG:
            # Tạo đường dẫn đầy đủ của tệp nguồn và đích
            source_path = os.path.join(source_folder, filename)
            target_path = os.path.join(target_folder, filename)
            # Sao chép tệp từ thư mục nguồn sang thư mục đích
            shutil.copyfile(source_path, target_path)
    print("Sao chép ảnh thành công trong 1 folder.")
def copy_in_many_folder_to_one_folder(source_folder, target_folder):
    # Tạo thư mục mới nếu chưa tồn tại
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # Duyệt qua tất cả các thư mục trong thư mục gốc
    for folder_name in os.listdir(source_folder):
        source_folder_path = os.path.join(source_folder, folder_name)
        # Kiểm tra nếu đối tượng là một thư mục
        if os.path.isdir(source_folder_path):
            # Duyệt qua tất cả các tệp trong thư mục con
            for filename in os.listdir(source_folder_path):
                # Tách tên tệp và phần mở rộng
                name, ext = os.path.splitext(filename)
                # Kiểm tra nếu tệp là ảnh và tên chứa một trong các chữ cái từ A-Z và số thứ tự từ 1 đến 1000
                if ext.lower() == '.jpg' and name[0].isalpha() and name[1:].isdigit() and int(name[1:]) <= NUMBER_IMG:
                    # Tạo đường dẫn đầy đủ của tệp nguồn và đích
                    source_path = os.path.join(source_folder_path, filename)
                    target_path = os.path.join(target_folder, filename)
                    # Sao chép tệp từ thư mục nguồn sang thư mục đích
                    shutil.copyfile(source_path,target_path)
    print("Sao chép ảnh thành công trong nhiều folder vào 1 folder.")
def copy_in_many_folder_to_many_folder(source_folder, target_folder):
    # Tạo thư mục mới nếu chưa tồn tại
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    # Duyệt qua tất cả các thư mục trong thư mục gốc
    for folder_name in os.listdir(source_folder):
        source_folder_path = os.path.join(source_folder, folder_name)
        # Kiểm tra nếu đối tượng là một thư mục
        if os.path.isdir(source_folder_path):
            # Duyệt qua tất cả các tệp trong thư mục con
            for filename in os.listdir(source_folder_path):
                # Tách tên tệp và phần mở rộng
                name, ext = os.path.splitext(filename)
                # Kiểm tra nếu tệp là ảnh và tên chứa một trong các chữ cái từ A-Z và số thứ tự từ 1 đến 1000
                if ext.lower() == '.jpg' and name[0].isalpha() and name[1:].isdigit() and int(name[1:]) <= NUMBER_IMG:
                    # Tạo đường dẫn đầy đủ của tệp nguồn và đích
                    source_path = os.path.join(source_folder_path, filename)
                    # kiểm tra xem đã có folder con tồn tại hay chưa
                    target_path = os.path.join(target_folder, folder_name)
                    if not os.path.exists(target_path):
                        os.makedirs(target_path)
                    target_path = os.path.join(target_folder, folder_name,filename)
                    # Sao chép tệp từ thư mục nguồn sang thư mục đích
                    shutil.copyfile(source_path, target_path)
    print("Sao chép ảnh thành công trong nhiều folder vào nhiều folder.")

copy_in_many_folder_to_one_folder(source_folder,target_folder)