# Comment classification 
Phân loại bình luận film, phục vụ cho việc đánh giá film

Sử dụng : 
* sử dụng môi trường python 3.6 để chạy
* sử dụng pip để cài 1 số gói cần thiết
 pip install -r requirements.txt 
* Để thực hiên kiểm tra độ chính xác của thuật toán với bộ test cho trước,
 Vào file app.py trong package Application, uncomment dòng thứ 6 và chạy file app.py
  ```python 
    # Train_Test.test()
  ```
* Lấy comment trực tiếp từ trên IMDB
  * Mở file src/application/App.py
  * Thay thuộc tính link của biến crawler bằng link reviews phim mong muốn, lưu ý link film phải là link trên trang IMDB.
  Link nhận được bằng cách nhấn vào phần Reviews ở mỗi film. <br> 
  VD : 
     ```python 
     crawler = SeleniumCrawler().run_crawler(link="https://www.imdb.com/title/tt5523010/reviews?ref_=tt_ov_rt")
     ```
  * Chạy file App.py
  * Cần phải cài selenium, beautifulsoup4, lxml, Chorme Driver để sử dụng 
* Thử với comment có sẵn
  * Cần 1 folder lưu các comment có sẵn bằng tiếng anh, mỗi file text viết 1 comment
  * Có thể thử với các folder có sẵn trong folder Data/MyData/neg hoặc Data/MyData/pos, đấy là 2 folder đã được crawl từ trước bằng cách bên trên 
  * Xóa (hoặc comment) dòng crawler = ... và thay thuộc tính folder_path bằng folder chứa comment.<br>
  VD : 
    ```python
    #crawler = SeleniumCrawler().run_crawler(link="https://www.imdb.com/title/tt5523010/reviews?ref_=tt_ov_rt")
    feature = FeatureFileBuilder(folder_path="../../Data/MyData/neg").build_feature_from_folder()
    ```
  * Chạy App.py, không cần cài thêm thư viện nào

Chương trình sau khi chạy sẽ hiển thị số lượng comment là positive hay là negative
