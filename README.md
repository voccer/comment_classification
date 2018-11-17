comment_classification 

<strong>MAKE FEATURE</strong><br>
Xây dựng dictionary + feature file nếu các file đó chưa tồn tại.<br>
File dictionary liệt kê tất cả các từ xuất hiện trong tập test, train mỗi từ xuất hiện trên 1 dòng, không sắp xếp thứ tự, dùng để ánh xạ các từ trong đoạn văn về dạng vecto BoW. <br>
Folder feature chứa tập các folder có dạng folder_number. Nghĩa là folder này chứa number file test_neg, test_pos, train_neg, train_pos mỗi loại. Dùng để test khi chương trình chạy với các số lượng file train, test khác nhau. <br>

<strong>DictionaryBuilder</strong><br>
Sử dụng chính file test, train để làm dictionary. Dùng thư viện nltk.tokenizer để tách từ trong các file train, test. Sau đó đưa vào set để lọc các từ trùng nhau rồi ghi vào file dictionary. Do set sử dụng cấu trúc hash nên tốc độ lọc từ trùng nhau là O1.<br>

<strong>FeatureFileBuilder</strong><br>
Dùng để xây dựng các file feature của file test, train sử dụng phương pháp BoW. Mỗi file test, train sẽ được biểu diễn bằng 1 tập n dòng, n là số từ xuất hiện trong file. Ở mỗi dòng, sẽ có 3 số : thứ tự file, thứ tự của từ đang xét trong từ điển, số lượng từ đó xuất hiện trong file. <br>
Để xây dựng các file feature, trước hết ta đọc dictionary vào 1 set để giảm thời gian tìm kiếm. Đọc lần lượt các path dẫn đến mỗi file train, test rồi đưa vào array để xử lý lần lượt. Với từng file train, test riêng rẽ, ta dùng nltk.tokenizer để tách thành các từ riêng. Sau đó tìm kiếm chúng trong dictionary. Lưu kết quả tìm được ở dạng Dictionary với Key là chỉ số xuất hiện của nó trong dictionay, Value là số lần từ đó xuất hiện trong file. Sau bước này ra sẽ được file feature của từng file train, test.

<strong>FileHandler</strong><br>
Chứa các hàm thao tác với file như read, write... 