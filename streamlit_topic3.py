#Nhập các thư viện cần thiết
import streamlit as st
import pickle  #it is a part of Python library
from pickle import load
#import numpy as np
import pandas as pd

import gensim
from gensim import corpora, models, similarities


##################################################
#Lọc ra 5 sản phẩm có (giá bán chưa giảm) 'list_price' cao nhất
#Load 'product_highest_list_price.csv'
product_highest_list_price = pd.read_csv('product_highest_list_price.csv', index_col=0)

#Lọc ra 5 sản phẩm có (giá bán chưa giảm) 'list_price' thấp nhất
#Load 'product_lowest_list_price.csv'
product_lowest_list_price = pd.read_csv('product_lowest_list_price.csv', index_col=0)

############### Thuật toán Gensim #######################


#Xem nội dụng phần tử đầu tiên trong cột 'name_description' khi chưa xử lý
with open('name_description_first_row_raw_content.txt', 'r', encoding='utf-8') as file:
    name_description_content = file.read()

#Lấy nội dụng phần tử đầu tiên trong cột 'name_description'
#sau khi xử lý văn bản thô, chuẩn hóa Unicode, word tokenize
with open('name_description_first_row_processed_content.txt', 'r', encoding='utf-8') as file:
    name_description_processed_content = file.read()


#Load tập tin product_df đã xử lý xong các bước trên
processed_product_df1 = pd.read_csv('processed_product_df1.csv', index_col=0)
processed_product_df2 = pd.read_csv('processed_product_df2.csv', index_col=0)
processed_product_df = pd.concat([processed_product_df1, processed_product_df2])

#Load bộ dictionary cho toàn bộ văn bản
gensim_dictionary = gensim.corpora.dictionary.Dictionary.load('gensim_model/gensim_dictionary')

#Load 'tfidf' được tính từ corpus
gensim_tfidf = gensim.models.tfidfmodel.TfidfModel.load('gensim_model/gensim_tfidf')

#Load 'index' - chứa các thông số tương tự của toàn bộ văn bản
gensim_index = gensim.similarities.docsim.SparseMatrixSimilarity.load('gensim_model/gensim_index')

#Đề xuất cho sản phẩm đang xem theo Gensim
#Định nghĩa hàm đề xuất sản phẩm tương tự với sản phẩm đang xem
@st.cache
def gensim_recommender(view_product, dictionary, tfidf, index):
    #Convert search words into Sparse Vectors
    view_product = view_product.lower().split()
    kw_vector = dictionary.doc2bow(view_product)
    print("View product's vector:")
    print(kw_vector)
    #Similarity calculation
    sim = index[tfidf[kw_vector]]
    
    #print result
    list_id = []
    list_score = []
    for i in range(len(sim)):
        list_id.append(i)
        list_score.append(sim[i])
        
    df_result = pd.DataFrame({'id': list_id, 'score': list_score})
    
    #Chọn ra 5 sản phẩm có score cao nhất
    five_highest_scores = df_result.sort_values('score', ascending=False).head(6)
    print('Five highest scores:')
    print(five_highest_scores)
    print('Ids to list:')
    idToList = list(five_highest_scores['id'])
    print(idToList)
    
    products_find = processed_product_df[processed_product_df.index.isin(idToList)]
    results = products_find[['index','item_id','name','price']]
    results = pd.concat([results, five_highest_scores], axis=1).sort_values('score',ascending=False)
    return results
    
    
################# Thuật toán Cosine Similarity ############################

#Load tập tin example của mô hình 'Cosine Similarity' 
cosine_similarities_df = pd.read_csv('cosine_similarities_df_example.csv', index_col=0)

#Định nghĩa hàm lấy thông tin sản phẩm
def item(product_id):
    return processed_product_df.loc[processed_product_df['item_id'] == product_id]['name'].tolist()[0].split('-')[0]

#Load mô hình 'Cosine Similarity'
cosine_results = load(open('cosine_model.pkl', 'rb'))

#Định nghĩa hàm cung cấp thông tin của các sản phẩm gợi ý theo Cosine Similarity
#Dưới dạng product_id và name tách riêng từng cột
def cosine_recommender(product_id, num):
    list_id = []
    list_name = []
    list_price = []
    recs = cosine_results[product_id][:num]
    for rec in recs:
        #recommended_text = 'Product_Id: ' + str(rec[1]) +' - ' + item(rec[1]) + ' (score:' + str(rec[0]) + ")"
        list_id.append(rec[1])
        list_price.append(processed_product_df[processed_product_df.item_id==rec[1]]['price'].values[0])
        list_name.append(item(rec[1]))
    
    rec_product_df = pd.DataFrame({'item_id': list_id, 'product_name': list_name, 'price': list_price})

    #list_text_df = pd.DataFrame(list_text, columns=['Các sản phẩm tương tự'])

    return rec_product_df



##################################################################
#avg_rating_customer.to_csv('avg_rating_customer.csv')
avg_rating_customer = pd.read_csv('avg_rating_customer.csv', index_col=0)


########### Phần tạo giao diện #############################################
#GUI
st.title('Data Science Project - Recommendation System')


#___________________________
#GUI
menu = ['Mục Tiêu Kinh Doanh','Khám Phá Dữ Liệu', 'Xây Dựng Mô Hình', 'Thực Hiện Đề Xuất']
choice = st.sidebar.selectbox('Menu',menu)
if choice == 'Mục Tiêu Kinh Doanh':
    st.subheader('Mục Tiêu Kinh Doanh')
    st.markdown("""
    - Tiki là 1 hệ sinh thái thương mại 'all in one', trong đó có tiki.vn, là một website thương mại điện tử đứng top 2 của Việt Nam, top 6 khu vực Đông Nam Á. Trên trang này đã triển khai nhiều tiện ích hỗ trợ nâng cao trải nghiệm người dùng và họ muốn xây dựng nhiều tiện ích hơn nữa.
    - Giả sử công ty này chưa triển khai Recommender System và bạn được yêu cầu triển khai hệ thống này, bạn sẽ làm gì?
    - Mục tiêu/vấn đề: xây dựng Recommendation System cho một hoặc một số nhóm hàng hóa trên tiki.vn giúp đề xuất và gợi ý cho người dùng/khách hàng.
    """)
    st.subheader('Triển khai dự án')
    st.markdown("""
    Chúng tôi nghiên cứu, khám phá dữ liệu công ty đang có và xây dựng mô hình theo 2 phương pháp:
    - Content-based filtering
    - Collaborative-based filtering
    """)
    st.image('images/Recommendation System Methods.png')


elif choice == 'Khám Phá Dữ Liệu':
    st.subheader('Tìm Hiểu và Khám Phá Dữ Liệu')
    st.markdown("""
    Trong quá trình tìm hiểu khám phá dữ liệu, chúng tôi đã rút ra được những thông tin hữu ích về sản phẩm, thương hiệu, khách hàng, giá cả và đánh giá chất lượng trên tiki.vn.
    """)
    
    #Top 5 sản phẩm có (giá bán chưa giảm) 'list_price' cao nhất
    product_highest_list_price = product_highest_list_price[['item_id','name','rating','price']]
    st.write("###### Top 5 sản phẩm có (giá bán chưa giảm) 'list_price' cao nhất:""")
    st.dataframe(product_highest_list_price.head(5))

    #Top 5 sản phẩm có (giá bán chưa giảm) 'list_price' thấp nhất
    product_lowest_list_price = product_lowest_list_price[['item_id','name','rating','price']]
    st.write("###### Top 5 sản phẩm có (giá bán chưa giảm) 'list_price' thấp nhất:""")
    st.dataframe(product_lowest_list_price.head(5))

    #Top 10 thương hiệu có số mã sản phẩm nhiều nhất
    st.write("###### Top 10 thương hiệu có số mã sản phẩm nhiều nhất:""")
    st.image('images/Number_of_products_by_brand.jpg')
    st.markdown(""" Nhận xét:
    - Số lượng sản phẩm của Samsung là vượt trội so với các hãng còn lại. Điều đó có nghĩa là Tiki đang đẩy mạnh triển khai thương hiệu Samsung với các mặt hàng sản phẩm đa dạng và phong phú.
    - Các hãng còn lại thì số lượng các mã sản phẩm là khá tương đồng, không chênh lệch nhiều.
    """)

    #Giá bán 'price' bình quân theo các thương hiệu
    st.write("###### Giá bán 'price' bình quân theo các thương hiệu:""")
    st.image('images/Average_price_by_brand.jpg')
    st.markdown(""" Nhận xét:
    - Giá bán bình quân của Hitachi là cao nhất và của Fujifilm là nhỏ nhất.
    - Giá bán bình quân của 4 hãng Fujifilm, Mitsubishi, Arber, Sigma là chênh lệch ít và thuộc nhóm giá thấp nhất.
    - Giá bán bình quân của 4 hãng Apple, BlackShark, Bosch, Surface cũng xấp xỉ nhau và thuộc nhóm giá cao, chỉ đứng sau Hitachi.
    - Chi riêng giá bán bình quân của hãng Acer là ở mức trung bình của top 10 thương hiệu.
    """)

    #Giá bán 'list_price' bình quân theo từng 'group' của các thương hiệu
    st.write("###### Giá bán 'list_price' bình quân theo từng 'group' của các thương hiệu:""")
    st.image('images/List_price_per_group_by_brand.jpg')
    st.markdown(""" Nhận xét:
    - Giá bán bình quân 'list_price' theo các nhóm hàng của hãng LG là cao nhất, kế đó là Microsoft.
    - Nhóm hàng có giá trị cao nhất là 'Điện tử-Điện Lạnh/Tivi/Tivi OLED và đứng thứ 2 là nhóm hàng 'Điện Thoại-Máy Tính Bảng/Máy tính bảng'.
    """)

    #Đánh giá 'rating' cho các sản phẩm
    st.write("###### Đánh giá 'rating' cho các sản phẩm:""")
    st.image('images/ratings_in_review_file.jpg')
    st.markdown(""" Nhận xét:
    - Phần lớn các sản phẩm được phản hồi khá là tích cực, có rating >= 4.0 trở lên. Điều đó chứng tỏ phần lớn khách hàng ưa chuộng về các sản phẩm.
    - Số lượng sản phẩm có rating bằng 5 là nhiều nhất.
    """)

    #Top 5 sản phẩm có rating bình quân cao nhất
    st.write("###### Top 5 sản phẩm có rating bình quân cao nhất (thích nhất):""")
    st.dataframe(avg_rating_customer.head(5))

    #Top 5 sản phẩm có rating bình quân thấp nhất
    st.write("###### Top 5 sản phẩm có rating bình quân thấp nhất:""")
    st.dataframe(avg_rating_customer.tail(5))

    #20 sản phẩm được khách hàng đánh giá nhiều nhất
    st.write("###### 20 sản phẩm được khách hàng đánh giá nhiều nhất:""")
    st.image('images/Products_highest_comments.jpg')
    st.markdown(""" Nhận xét:
    - Sản phẩm được đánh giá nhiều nhất là 'Chuột Không Dây Logitech' với số lượng đánh giá nhiều hơn hẳn so với các sản phẩm còn lại.
    - Các sản phẩm còn lại thì có số lượng đánh giá giảm dần đều với sự chênh lệch không đáng kể.
    """)

    #20 khách hàng thực hiện đánh giá sản phẩm nhiều nhất
    st.write("###### 20 khách hàng thực hiện đánh giá sản phẩm nhiều nhất:""")
    st.image('images/Customers_rating_the_most.jpg')
    st.markdown(""" Nhận xét:
    Số lần đánh giá sản phẩm nhiều nhất mà 1 khách hàng thực hiện là 50 lần cho 50 sản phẩm, và thấp nhất là 25 lần cho 25 sản phẩm trong top 20 khách hàng đứng đầu.
    """)

elif choice == 'Xây Dựng Mô Hình':
    #Bài toán 1: Xây dựng model theo content-based filtering
    st.write("""#### 1. Xây dựng Recommendation System theo content-based filtering
    """)
    st.markdown("""
    Đối với phương pháp này, cụ thể là Gensim và Cosine Similarity, chúng tôi xây dựng mô hình Recommendation System theo quy trình sau:
    - Sử dụng thông tin tổng hợp ‘name_description’ của tên sản phẩm và phần miêu tả sản phẩm. 
    - Thông tin này kế tiếp sẽ được xử lý làm sạch, chuẩn hóa và tách thành các từ riêng biệt tạo thành bộ từ điển của sản phẩm. 
    - Sau đó, các từ trong bộ từ điển này sau đó sẽ được tính toán mức độ tương đồng lẫn nhau và được số hóa tạo thành một bộ đo lường tiêu chuẩn.
    - Với thông tin tên và miêu tả của một sản phẩm bất kỳ được cung cấp, bộ đo lường này sẽ được sử dụng để đưa ra các đề xuất/gợi ý cho các sản phẩm tương tự khác.
    """)
    
    st.write("###### Minh họa thông tin tổng hợp 'name_description' của 1 sản phẩm khi chưa xử lý:""")
    st.markdown(name_description_content)

    st.write("""###### Và sau khi được xử lý làm sạch, chuẩn hóa nội dung:""")
    st.markdown(name_description_processed_content)

    #1.1 Sử dụng thuật toán Gensim
    st.write("##### 1.1 Sử dụng thuật toán Gensim")

    st.write("""###### Source code xây dựng bộ đo lường/tính toán tiêu chuẩn 'index' theo Gensim:""")
    tfidf_index_code = """
    #Phân rã nội dung ‘name_description’ thành các từ riêng biệt
    intro_products = [[text for text in x.split()] for x in product_df['name_description_wt']]
    #Loại bỏ stop_words
    intro_products_remove = [[t for t in text if not t in stop_words] for text in intro_products]
    #Tạo bộ từ điển của các từ
    dictionary = corpora.Dictionary(intro_products_remove)
    #Tính số từ trong bộ từ điển ‘dictionary’
    feature_cnt = len(dictionary.token2id)
    #Tính số lần xuất hiện của từng từ trong bộ dictionary
    corpus = [dictionary.doc2bow(text) for text in intro_products_remove]
    #Sử dụng TF-IDF để tính trọng số cho các từ
    tfidf = models.TfidfModel(corpus)
    #Tạo bộ đo lường/tính toán tiêu chuẩn mức độ tương tự của các từ
    index = similarities.SparseMatrixSimilarity(tfidf[corpus],num_features = feature_cnt)
    """
    st.code(tfidf_index_code, language='Python')

    #Đề xuất 5 sản phẩm tương tự sản phẩm đang xem theo Gensim
    st.write("""###### Giả sử sản phẩm đang xem có product_ID = 10001355, 5 sản phẩm tương tự được đề xuất:""")
    gensim_example = pd.read_csv('gensim_example.csv', index_col=0)
    st.dataframe(gensim_example)

    #1.2 Sử dụng thuật toán Cosine Similarity
    st.write("""##### 1.2 Sử dụng thuật toán Cosine Similarity""")
    
    st.write("""###### Source code xây dựng bộ đo lường/tính toán tiêu chuẩn theo Cosine Similarity:""")
    tfidf_cosine_code = """
    #Tạo object Tfidf Vectorizer
    tfidf_vec = TfidfVectorizer(analyzer='word', min_df=0, stop_words = stop_words)
    #Thực hiện transformation cho cột văn bản đã xử lý làm sạch và chuẩn hóa
    tfidf_matrix = tfidf_vec.fit_transform(processed_product_df['name_description_wt'])
    #Tạo bộ đo lường/tính toán tiêu chuẩn mức độ tương tự cho toàn bộ các từ trong văn bản
    cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)
    """
    st.code(tfidf_cosine_code, language='Python')

    st.write("""###### Minh họa vài dòng đầu tiên của bộ đo lường/tính toán 'cosine_similarities':""")
    st.dataframe(cosine_similarities_df.head(5))

    #Đề xuất 5 sản phẩm tương tự sản phẩm đang xem theo cosine_similarity
    st.write("""###### Giả sử sản phẩm đang xem có product_ID = 1059892, 5 sản phẩm tương tự được đề xuất:""")
    recommended_products = pd.read_csv('cosine_similarity_example.csv', index_col=0)
    st.table(recommended_products)

    #1.3 So sánh 2 thuật toán Gensim và Cosine Similarity
    st.write("""#### 1.3 So sánh 2 thuật toán""")
    st.markdown("""
    - Cả hai model theo Gensim và Cosine_similarity đểu cho ra kết quả rất tốt về đề xuất các sản phẩm tương tự với sản phẩm khách hàng đang xem.
    - Gensim thì có thời gian tính toán nhanh hơn 1 chút so với Cosine_similarity. Tuy nhiên, Cosine_similarity thì lại cho ra kết quả các sản phẩm có mức độ tương tự với sản phẩm đang xem tốt hơn so với Gensim.
    """)

    #Bài toán 2: Xây dựng model theo collaborative-based filtering
    st.write("""#### 2. Xây dựng model theo collaborative-based filtering""")
    #Sử dụng thuật toán ALS
    st.markdown("""
    Đối với phương pháp này, chúng tôi xây dựng mô hình đề xuất/gợi ý theo quy trình sau:
    - Sử dụng 3 thông tin quan trọng là: mã khách hàng, mã sản phẩm và đánh giá.
    - Áp dụng thuật toán ALS (Alternating Least Square) để tạo mô hình huấn luyện và đánh giá độ chính xác theo RMSE.
    - Với thông tin của một mã khách hàng bất kỳ được cung cấp, mô hình ALS được sử dụng để đề xuất/gợi ý các sản phẩm tương tự khác.
    """)

    st.write("""###### Source code khởi tạo mô hình ALS và huấn luyện mô hình:""")
    ALS_train_code ="""
    #Chia bộ dữ liệu theo tỉ lệ 80:20
    (training, test) = data_sub.randomSplit([0.8,0.2])
    #Khởi tạo ALS model và điều chỉnh các tham số thích hợp
    als = ALS(maxIter=20,regParam=0.5,alpha=0.5,
          rank=5,userCol='customer_id',itemCol='product_id',
          ratingCol='rating',coldStartStrategy='drop',nonnegative=True)
    #Huấn luyện mô hình
    model = als.fit(training)
    """
    st.code(ALS_train_code, language='Python')

    st.write("###### Source code đánh giá mô hình theo RMSE:")
    rmse_code = """
    #Tính các giá trị dự đoán theo bộ test
    predictions = model.transform(test)
    #Tính RMSE để đánh giá mô hình
    evaluator = RegressionEvaluator(metricName='rmse',labelCol='rating',predictionCol='prediction')
    rmse = evaluator.evaluate(predictions)
    """
    st.code(rmse_code, language='Python')

    #Đề xuất 10 sản phẩm có rating cao nhất cho 1 khách hàng cụ thể
    st.write("###### Giả sử có khách hàng customer_id = 6177374 đang xem sản phẩm, top 10 sản phẩm được đề xuất:""")
    st.image('images/recommended_products_specific_customer.jpg')


elif choice == 'Thực Hiện Đề Xuất':
    model_select = st.sidebar.radio('Chọn phương pháp:',options=['Content_Filtering','Collaborative_Filtering'])
    if model_select == 'Content_Filtering':
        
        sample_products = pd.read_csv('sample_products_list.csv', index_col=0)
        sample_products_df = sample_products.values.reshape(3,5)
        sample_products_df = pd.DataFrame(sample_products_df)

        st.write("""### 1. Đề xuất sản phẩm tương tự theo content_filtering""")
        st.write("""###### Chọn 1 trong 15 sản phẩm minh họa trong bảng danh mục Product_ID bên dưới:""")
        st.table(sample_products_df)

        #Nhập chọn 1 product_id, mã product_id mặc định là chọn mã đầu tiên trong bảng
        product_id_input = st.text_input('Chọn một Product_ID trong bảng:', value=sample_products['product_id'][0])
        
        algorithm_input = st.selectbox('Algorithm', options=['Gensim','Cosine_Similarity'])
        
        submit = st.button('Recommend')
        if product_id_input=="":
            st.write("##### Vui lòng nhập vào 1 mã sản phẩm ở trong bảng!""")
        elif product_id_input is not None:
            if submit:
                
                st.write("""##### Sản phẩm được chọn:""")
                product_name = item(int(product_id_input))
                sample_price = processed_product_df[processed_product_df.item_id == int(product_id_input)]['price'].values[0]
                sample_price = 'Price: ' + str(sample_price) + ' VND'

                image_site = 'gensim_rec_prod_img/' + str(product_id_input) +'.jpg'

                st.text(product_name)
                st.markdown(sample_price)
                st.image(image_site)
                st.write("""##### Đề xuất/gợi ý bên dưới:""")

                if algorithm_input == 'Gensim':
                    view_product = processed_product_df[processed_product_df.item_id == int(product_id_input)].head(1)
                    name_description_wt = view_product['name_description_wt'].to_string(index=False)
                    results = gensim_recommender(name_description_wt, gensim_dictionary, gensim_tfidf, gensim_index)
                    results = results[results.item_id != int(product_id_input)]
                    results = results[['item_id','name','price']].reset_index()
                    results = results.drop('index',axis=1)
                    rec_prod_num = len(results)
                    for i in range(rec_prod_num):
                        col1, col2 = st.columns(2)
                        with col1:
                            image_link = 'gensim_rec_prod_img/' + str(results['item_id'][i]) +'.jpg'
                            st.image(image_link)
                        with col2:
                            product_id = 'Product ID: ' + str(results['item_id'][i])
                            prod_name = results['name'][i]
                            product_price = 'Price: ' + str(results['price'][i]) + ' VND'
                            st.subheader(product_id)
                            st.markdown(prod_name)
                            st.markdown(product_price)


                else:
                    sim_products = cosine_recommender(int(product_id_input), 5)
                    rec_prod_num = len(sim_products)
                    for i in range(rec_prod_num):
                        col1, col2 = st.columns(2)
                        with col1:
                            image_link = 'cosine_rec_prod_img/' + str(sim_products['item_id'][i]) +'.jpg'
                            st.image(image_link)
                        with col2:
                            product_id = 'Product ID: ' + str(sim_products['item_id'][i])
                            prod_name = sim_products['product_name'][i]
                            product_price = 'Price: ' + str(sim_products['price'][i]) + ' VND'
                            st.subheader(product_id)
                            st.markdown(prod_name)
                            st.markdown(product_price)


    elif model_select == 'Collaborative_Filtering':
        #Load mô hình ALS
        #ALS_recommender_model = load(open('ALS_model.pkl','rb')) #The size > 25MB on Github
        ALS_recommender_model1 = pd.read_csv('ALS_model/ALS_model_df_part1.csv',index_col=0)
        ALS_recommender_model2 = pd.read_csv('ALS_model/ALS_model_df_part2.csv',index_col=0)
        ALS_recommender_model3 = pd.read_csv('ALS_model/ALS_model_df_part3.csv',index_col=0)

        #Nối các file lại theo dòng 
        ALS_recommender_model = pd.concat([ALS_recommender_model1,
                                        ALS_recommender_model2, ALS_recommender_model3])

        #Load tập tin mẫu về customer_id
        sample_customers_df = pd.read_csv('ALS_customer_id_examples.csv', index_col=0)

        st.write("""### 2. Đề xuất sản phẩm tương tự theo collaborative_filtering""")
        st.write("""###### Chọn 1 trong 40 khách hàng minh họa trong bảng danh mục Customer_ID bên dưới:""")
        st.dataframe(sample_customers_df)

        #Nhập chọn 1 customer_id trong bảng, mặc định lấy giá trị đầu tiên trong bảng
        customer_id_input = st.text_input('Chọn một Customer_ID trong bảng:', value=sample_customers_df.iloc[0][0])
        algorithm_input = st.selectbox('Algorithm', options=['ALS'])
        
        submit = st.button('Recommend')

        if customer_id_input=="":
            st.write("##### Vui lòng nhập vào 1 mã khách hàng ở trong bảng!""")
        elif customer_id_input is not None:
            if submit:
                    customer_id = int(customer_id_input)
                    st.write("""##### Các sản phẩm đề xuất/gợi ý bên dưới:""")

                    if algorithm_input == 'ALS':
                        result = ALS_recommender_model[ALS_recommender_model['customer_id']==customer_id].reset_index()
                        result = result.drop('index',axis=1)
                        result = result[['product_id','rating']].head(5)
    
                        rec_prod_num = len(result)
                        for i in range(rec_prod_num):
                            col1, col2 = st.columns(2)
                            with col1:
                                image_link = 'ALS_rec_prod_img/' + str(result['product_id'][i]) +'.jpg'
                                st.image(image_link)
                            with col2:
                                product_id = 'Product ID: ' + str(result['product_id'][i])
                                prod_name = item(result['product_id'][i])
                                product_price = processed_product_df[processed_product_df.item_id == result['product_id'][i]]['price'].values[0]
                                product_price = 'Price: ' + str(product_price) + ' VND'
                                st.subheader(product_id)
                                st.markdown(prod_name)
                                st.markdown(product_price)
             




