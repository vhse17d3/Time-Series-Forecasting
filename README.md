<style>
  table {
    border-collapse: collapse; /* Hợp nhất các đường viền của ô */
    width: auto; /* Tự động điều chỉnh kích thước bảng */
    margin: auto; /* Canh giữa bảng */
  }
  th, td {
    border: 1px solid black; /* Đường viền cho mỗi ô */
    padding: 8px; /* Thêm khoảng cách giữa nội dung và đường viền của ô */
    text-align: left; /* Canh lề trái cho nội dung trong cột */
  }
</style>
# What is a Time Series?
- What is a Time Series?
Đối tượng cơ bản của dự báo là Time series, là một tập hợp các quan sát được ghi lại theo thời gian. Trong các ứng dụng dự báo, các quan sát thường được ghi lại với tần suất đều đặn, như hàng ngày hoặc hàng tháng.


In [1]:
``` python
import pandas as pd

df = pd.read_csv(
    "../input/ts-course-data/book_sales.csv",
    index_col='Date',
    parse_dates=['Date'],
).drop('Paperback', axis=1)

df.head()
```
Out[1]:
<table>
  <tr>
    <th>Date</th>
    <th>Hardcover</th>
  </tr>
  <tr>
    <td>2000-04-01</td>
    <td>139</td>
  </tr>
  <tr>
    <td>2000-04-02</td>
    <td>128</td>
  </tr>
  <tr>
    <td>2000-04-03</td>
    <td>172</td>
  </tr>
  <tr>
    <td>2000-04-04</td>
    <td>139</td>
  </tr>
  <tr>
    <td>2000-04-05</td>
    <td>191</td>
  </tr>
</table>




Bộ sách này ghi lại số lượng bán sách bìa cứng tại một cửa hàng bán lẻ trong 30 ngày. Lưu ý rằng chúng ta có một cột quan sát duy nhất Hardcover với chỉ mục thời 
gian Date.
## Linear Regression with Time Series
* Đối với phần đầu tiên của khóa học này, chúng ta sẽ sử dụng thuật toán hồi quy tuyến tính để xây dựng các mô hình dự báo. Hồi quy tuyến tính được sử dụng rộng rãi trong thực tế và thích ứng tự nhiên với các nhiệm vụ dự báo phức tạp.

Các **linear regression** Thuật toán học cách tạo tổng trọng số từ các tính năng đầu vào của nó. Đối với hai tính năng, chúng tôi sẽ có:
* **target = weight_1 * feature_1 + weight_2 * feature_2 + bias**

Trong quá trình đào tạo, thuật toán hồi quy học các giá trị cho các tham số weight_1, weight_2, và bias phù hợp nhất với target. (Thuật toán này thường được gọi là bình phương nhỏ nhất thông thường vì nó chọn các giá trị giảm thiểu sai số bình phương giữa mục tiêu và dự đoán.) Các trọng  số còn được gọi là hệ số hồi quy và độ lệch còn được gọi là chặn vì nó cho bạn biết nơi đồ thị của hàm này cắt trục y.
#### Time-step features
* Có hai loại tính năng duy nhất cho chuỗi thời gian: time-step features and lag features.
Tính năng bước thời gian là các tính năng chúng ta có thể lấy trực tiếp từ chỉ số thời gian. Tính năng bước thời gian cơ bản nhất là time dummy(hình nộm), đếm các bước thời gian trong chuỗi từ đầu đến cuối.

In [2]:
```python
import numpy as np

df['Time'] = np.arange(len(df.index))

df.head()
```
Out[2]:
<table>
  <thead>
    <tr>
      <th>Date</th>
      <th>Hardcover</th>
      <th>Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2000-04-01</td>
      <td>139</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2000-04-02</td>
      <td>128</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2000-04-03</td>
      <td>172</td>
      <td>2</td>
    </tr>
    <tr>
      <td>2000-04-04</td>
      <td>139</td>
      <td>3</td>
    </tr>
    <tr>
      <td>2000-04-05</td>
      <td>191</td>
      <td>4</td>
    </tr>
  </tbody>
</table>

Linear regression with the time dummy produces the model:
* **target = weight * time + bias**

Hình nộm thời gian sau đó cho phép chúng ta khớp các đường cong với chuỗi  thời gian trong một biểu đồ thời gian, trong đó Thời gian tạo thành trục x.

In [3]:
``` python
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-whitegrid")
plt.rc(
    "figure",
    autolayout=True,
    figsize=(11, 4),
    titlesize=18,
    titleweight='bold',
)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=16,
    titlepad=10,
)
%config InlineBackend.figure_format = 'retina'

fig, ax = plt.subplots()
ax.plot('Time', 'Hardcover', data=df, color='0.75')
ax = sns.regplot(x='Time', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_title('Time Plot of Hardcover Sales');
 ```

![alt text](https://www.kaggleusercontent.com/kf/126573838/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..wqM7b7PCPWdTGftYdv_Zmg.fRvM1H6RId81co9CCX2Y5LOX7rDv6W3h2kt654oUtlQqQAfB8xtB-835hBwSZZ98KNglcj1xH1Rvm2Onw3CrBs-RRXnPMj1LwBAf0VFOULjNrq-__-94u0sj2cJ8Fz1zyrHlTBZgdtA7vSnD8UheHD2CPK8g3SBpJzbAd07fIn7ZAXt4O87dQ_TVyjuqILTNjkwroTjZDhhNFv-ZuURDR9RgmLrYH7_YQqLTE-ncLy-DUtLHNXibwfsPAjR5CB0w8MVjyBeFYKTWCNeegeEik22sVi5-G72eYhS3JA2ZE3aI1RirpnpcgK3GFTrNYTLARU-hiSrpSeYEf5DoWN00kOKqojjHJ4Gn39pSaEm3bwZNRaGBonx1Q6kJwU3EfNIs0fsu2nYCUwYNuk5Glyg5bFb_9rzEVcF8rD2WDdtHiPEKoLknR8g8_iVXC-IZZdxbPl9ivOekcEPjhMfijHOd6BUyoNOilNydVYEdOd0xLi2iLmyhqQ0QeZrraKMYCdDFc1RyHy2vFrcZ86Ngf3Y0_8oyy3aXyvrPSIVfXJbRB-hkwMEhr49FJ1lj9QQpjVscA7veFUu29EY4mCAaqTTg9azrx8X39nZrhWIV1spOUHt8RbBNMih8Py4zRCLxSBEH4Pt8AnWOQFn0gamh2cYJXsCG2T3BdNw1_QHY8qMPLDN7L5aT7T5Jg0ytMWKhHfYu.cbF6bL8sZYEboHk6ISyLjA/__results___files/__results___5_0.png)

* Time-step Các tính năng cho phép bạn lập mô hình time dependence. Một chuỗi phụ thuộc vào thời gian nếu các giá trị của nó có thể được dự đoán từ thời điểm chúng xảy ra. Trong Hardcover Sales series, Chúng tôi có thể dự đoán rằng doanh số bán hàng vào cuối tháng thường cao hơn doanh số bán hàng vào đầu tháng.
#### Lag features
* Để tạo một **lag feature** Chúng ta thay đổi các quan sát của chuỗi mục tiêu để chúng dường như xảy ra muộn hơn. Ở đây chúng tôi đã tạo tính năng độ trễ 1 bước, mặc dù cũng có thể dịch chuyển theo nhiều bước.

In [4]:
```python
df['Lag_1'] = df['Hardcover'].shift(1)
df = df.reindex(columns=['Hardcover', 'Lag_1'])

df.head()
```
Out[4]:
<table>
  <thead>
    <tr>
      <th>Date</th>
      <th>Hardcover</th>
      <th>Lag_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>2000-04-01</td>
      <td>139</td>
      <td>NaN</td>
    </tr>
    <tr>
      <td>2000-04-02</td>
      <td>128</td>
      <td>139.0</td>
    </tr>
    <tr>
      <td>2000-04-03</td>
      <td>172</td>
      <td>128.0</td>
    </tr>
    <tr>
      <td>2000-04-04</td>
      <td>139</td>
      <td>172.0</td>
    </tr>
    <tr>
      <td>2000-04-05</td>
      <td>191</td>
      <td>139.0</td>
    </tr>
  </tbody>
</table>

Linear regression with a lag feature produces the model:
* **target = weight * lag + bias**

Vì vậy, các tính năng độ trễ cho phép chúng ta khớp các đường cong  với các biểu đồ trễ trong đó mỗi quan sát  trong một chuỗi được vẽ so với quan sát trước đó.

``` python
fig, ax = plt.subplots()
ax = sns.regplot(x='Lag_1', y='Hardcover', data=df, ci=None, scatter_kws=dict(color='0.25'))
ax.set_aspect('equal')
ax.set_title('Lag Plot of Hardcover Sales');
```
 

![alt text](https://www.kaggleusercontent.com/kf/126573838/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..wqM7b7PCPWdTGftYdv_Zmg.fRvM1H6RId81co9CCX2Y5LOX7rDv6W3h2kt654oUtlQqQAfB8xtB-835hBwSZZ98KNglcj1xH1Rvm2Onw3CrBs-RRXnPMj1LwBAf0VFOULjNrq-__-94u0sj2cJ8Fz1zyrHlTBZgdtA7vSnD8UheHD2CPK8g3SBpJzbAd07fIn7ZAXt4O87dQ_TVyjuqILTNjkwroTjZDhhNFv-ZuURDR9RgmLrYH7_YQqLTE-ncLy-DUtLHNXibwfsPAjR5CB0w8MVjyBeFYKTWCNeegeEik22sVi5-G72eYhS3JA2ZE3aI1RirpnpcgK3GFTrNYTLARU-hiSrpSeYEf5DoWN00kOKqojjHJ4Gn39pSaEm3bwZNRaGBonx1Q6kJwU3EfNIs0fsu2nYCUwYNuk5Glyg5bFb_9rzEVcF8rD2WDdtHiPEKoLknR8g8_iVXC-IZZdxbPl9ivOekcEPjhMfijHOd6BUyoNOilNydVYEdOd0xLi2iLmyhqQ0QeZrraKMYCdDFc1RyHy2vFrcZ86Ngf3Y0_8oyy3aXyvrPSIVfXJbRB-hkwMEhr49FJ1lj9QQpjVscA7veFUu29EY4mCAaqTTg9azrx8X39nZrhWIV1spOUHt8RbBNMih8Py4zRCLxSBEH4Pt8AnWOQFn0gamh2cYJXsCG2T3BdNw1_QHY8qMPLDN7L5aT7T5Jg0ytMWKhHfYu.cbF6bL8sZYEboHk6ISyLjA/__results___files/__results___9_0.png)

Bạn có thể thấy từ biểu đồ độ trễ rằng doanh số bán hàng trong một ngày **(Hardcover)** có tương quan với doanh số bán hàng từ ngày hôm trước **(Lag_1)**. Khi bạn thấy một mối quan hệ như thế này, bạn biết một tính năng độ trễ sẽ hữu ích.
Tổng quát hơn, các tính năng độ trễ cho phép bạn mô hình hóa sự phụ thuộc nối tiếp. Một chuỗi thời gian có sự phụ thuộc nối tiếp khi một quan sát có thể được dự đoán từ các quan sát trước đó. Trong Bán hàng Hardcover, chúng ta có thể dự đoán rằng doanh số bán hàng cao vào một ngày thường có nghĩa là doanh số bán hàng cao vào ngày hôm sau.
________________________________________
Điều chỉnh các thuật toán học máy cho các vấn đề chuỗi thời gian chủ yếu là về kỹ thuật tính năng với chỉ số thời gian và độ trễ. Đối với hầu hết các khóa học, chúng tôi sử dụng hồi quy tuyến tính vì tính đơn giản của nó, nhưng các tính năng này sẽ hữu ích cho bất kỳ thuật toán nào bạn chọn cho nhiệm vụ dự báo của mình.
## What is Trend?
* The trend Thành phần của chuỗi thời gian đại diện cho một sự thay đổi liên tục, lâu dài trong giá trị trung bình của chuỗi. Xu hướng này là phần chuyển động chậm nhất của một chuỗi, phần đại diện cho quy mô thời gian quan trọng lớn nhất. Trong một chuỗi thời gian bán sản phẩm, xu hướng ngày càng tăng có thể là ảnh hưởng của việc mở rộng thị trường khi nhiều người biết đến sản phẩm qua từng năm.
 
Trend patterns in four time series.
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/ZdS4ZoJ.png)
<center>
Trend patterns in four time series.
</center>

Chúng tôi sẽ tập trung vào các xu hướng trong giá trị trung bình. Nói chung, bất kỳ thay đổi liên tục và chậm chạp nào trong một chuỗi có thể tạo thành một xu hướng - ví dụ như chuỗi thời gian thường có xu hướng trong biến thể của chúng.
#### Moving Average Plots ( Biểu đồ trung bình động )
* Để xem loại xu hướng mà chuỗi thời gian có thể có, chúng ta có thể sử dụng biểu đồ trung bình động. Để tính đường trung bình động của một chuỗi thời gian, chúng tôi tính giá trị trung bình của các giá trị trong một cửa sổ trượt có độ rộng xác định nào đó. Mỗi điểm trên biểu đồ đại diện cho giá trị trung bình của tất cả các giá trị trong chuỗi nằm trong cửa sổ ở hai bên. Ý tưởng là làm dịu bất kỳ biến động ngắn hạn nào trong chuỗi để chỉ còn lại những thay đổi dài hạn.

![alt text](https://storage.googleapis.com/kaggle-media/learn/images/EZOXiPs.gif)
<center>
A moving average plot illustrating a linear trend. Each point on the curve (blue) is the average of the points (red) within a window of size 12.
</center> 

Lưu ý cách  loạt Mauna Loa ở trên có chuyển động lên xuống lặp đi lặp lại năm này qua năm khác - một sự thay đổi ngắn hạn, theo mùa. Để thay đổi trở thành một phần của xu hướng, nó nên xảy ra trong một khoảng thời gian dài hơn bất kỳ thay đổi theo mùa nào. Do đó, để hình dung một xu hướng, chúng tôi lấy trung bình trong một khoảng thời gian dài hơn bất kỳ khoảng thời gian theo mùa nào trong chuỗi. Đối với  dòng Mauna Loa, chúng tôi đã chọn một cửa sổ cỡ 12 để làm mịn theo mùa trong mỗi năm.
#### Engineering Trend
* Khi chúng tôi đã xác định được hình dạng của xu hướng, chúng tôi có thể cố gắng mô hình hóa nó bằng cách sử dụng tính năng bước thời gian. Chúng ta đã thấy cách sử dụng hình nộm thời gian sẽ mô hình hóa một xu hướng tuyến tính:
**target = a * time + b**

Chúng ta có thể phù hợp với nhiều loại xu hướng khác thông qua các biến đổi của hình nộm thời gian. Nếu xu hướng xuất hiện là bậc hai **(một parabol)**, chúng ta chỉ cần thêm bình phương của hình nộm thời gian vào bộ tính năng, cho chúng ta:
* **target = a * time ** 2 + b * time + c**

Linear regression sẽ tìm hiểu các hệ số a, b, and c.
Các đường cong xu hướng trong hình dưới đây đều phù hợp khi sử dụng các loại tính năng này và scikit-learnLinearRegression:
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/KFYlgGm.png)

<center> 
Top: Series with a linear trend. Below: Series with a quadratic trend.
</center>

Nếu bạn chưa từng thấy mẹo này trước đây, bạn có thể không nhận ra rằng hồi quy tuyến tính có thể phù hợp với các đường cong khác ngoài các đường. Ý tưởng là nếu bạn có thể cung cấp các đường cong có hình dạng thích hợp làm tính năng, thì hồi quy tuyến tính có thể học cách kết hợp chúng theo cách phù hợp nhất với mục tiêu.
## What is Seasonality?
* Chúng tôi nói rằng một chuỗi thời gian thể hiện tính thời vụ (Seasonality) bất cứ khi nào có sự thay đổi thường xuyên, định kỳ trong giá trị trung bình của chuỗi. Thay đổi theo mùa thường theo đồng hồ và lịch - lặp lại trong một ngày, một tuần hoặc một năm là phổ biến. Tính thời vụ thường được thúc đẩy bởi các chu kỳ của thế giới tự nhiên qua nhiều ngày và nhiều năm hoặc bởi các quy ước về hành vi xã hội xung quanh ngày và giờ.
 

![alt text](https://storage.googleapis.com/kaggle-media/learn/images/ViYbSxS.png )
<center> 
Seasonal patterns in four time series.
</center>

Chúng ta sẽ tìm hiểu hai loại tính năng mô hình hóa tính thời vụ. Loại đầu tiên, các chỉ số, là tốt nhất cho một mùa có ít quan sát, giống như một mùa quan sát hàng ngày hàng tuần. Loại thứ hai, tính năng Fourier, là tốt nhất cho một mùa có nhiều quan sát, giống như một mùa quan sát hàng ngày hàng năm.
#### Seasonal Plots and Seasonal Indicators
* Giống như chúng ta đã sử dụng biểu đồ trung bình động để khám phá xu hướng trong một chuỗi, chúng ta có thể sử dụng seasonal plot để khám phá các mô hình theo mùa.
Một biểu đồ theo mùa hiển thị các phân đoạn của chuỗi thời gian được vẽ dựa trên một số khoảng thời gian phổ biến, khoảng thời gian là "mùa" bạn muốn quan sát. Hình vẽ cho thấy một biểu đồ theo mùa về lượt  xem hàng ngày của bài viết Wikipedia về lượng giác: lượt xem hàng ngày của bài viết được vẽ trong một  khoảng thời gian hàng tuần chung.
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/bd7D4NJ.png )
<center> 
There is a clear weekly seasonal pattern in this series, higher on weekdays and falling towards the weekend.
</center>

#### Seasonal indicators
* Seasonal indicators là các tính năng nhị phân đại diện cho sự khác biệt theo mùa trong cấp độ của một chuỗi thời gian. Các chỉ số theo mùa là những gì bạn nhận được nếu bạn coi một khoảng thời gian theo mùa là một tính năng phân loại và áp dụng one-hot encoding.

By one-hot encoding Các ngày trong tuần, chúng tôi nhận được các chỉ số theo mùa hàng tuần. Tạo các chỉ số hàng tuần  cho chuỗi Lượng giác sau đó sẽ cung cấp cho chúng ta sáu tính năng "giả" mới. (Hồi quy tuyến tính hoạt động tốt nhất nếu bạn bỏ một trong các chỉ báo; chúng tôi đã chọn thứ Hai trong khung bên dưới.)

| Date       | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday |
|------------|---------|-----------|----------|--------|----------|--------|
| 2016-01-04 | 0.0     | 0.0       | 0.0      | 0.0    | 0.0      | 0.0    |
| 2016-01-05 | 1.0     | 0.0       | 0.0      | 0.0    | 0.0      | 0.0    |
| 2016-01-06 | 0.0     | 1.0       | 0.0      | 0.0    | 0.0      | 0.0    |
| 2016-01-07 | 0.0     | 0.0       | 1.0      | 0.0    | 0.0      | 0.0    |
| 2016-01-08 | 0.0     | 0.0       | 0.0      | 1.0    | 0.0      | 0.0    |
| 2016-01-09 | 0.0     | 0.0       | 0.0      | 0.0    | 1.0      | 0.0    |
| 2016-01-10 | 0.0     | 0.0       | 0.0      | 0.0    | 0.0      | 1.0    |
| 2016-01-11 | 0.0     | 0.0       | 0.0      | 0.0    | 0.0      | 0.0    |


Thêm các chỉ số theo mùa vào dữ liệu đào tạo giúp các mô hình phân biệt phương tiện trong một khoảng thời gian theo mùa:
 ![alt text](https://storage.googleapis.com/kaggle-media/learn/images/hIlF5j5.png)
<center> 
Ordinary linear regression learns the mean values at each time in the season.
</center>

Các chỉ báo hoạt động như công tắc Bật / Tắt. Bất cứ lúc nào, tối đa một trong các chỉ báo này có thể có giá trị là 1 (Bật). Hồi quy tuyến tính học một giá trị cơ sở 2379 cho Mon và  sau đó điều chỉnh theo giá trị của bất kỳ chỉ báo nào được On cho ngày đó; phần còn lại là 0 và biến mất. 
#### Fourier Features and the Periodogram
* Loại tính năng mà chúng ta thảo luận bây giờ phù hợp hơn cho các mùa dài qua nhiều quan sát trong đó các chỉ số sẽ không thực tế. Thay vì tạo ra một tính năng cho mỗi ngày, các tính năng của Fourier cố gắng nắm bắt hình dạng tổng thể của đường cong theo mùa chỉ với một vài tính năng.

Chúng ta hãy xem một cốt truyện cho mùa giải hàng năm trongTrigonometry. Chú ý sự lặp lại của các tần số khác nhau: một chuyển động lên xuống dài ba lần một năm, chuyển động ngắn hàng tuần 52 lần một năm và có lẽ những tần số khác.
  ![alt text](https://storage.googleapis.com/kaggle-media/learn/images/NJcaEdI.png )
<center> 
Annual seasonality in the Wiki Trigonometry series.
</center>

Đó là những tần số này trong một mùa mà chúng tôi cố gắng nắm bắt Fourier features. Ý tưởng là đưa vào dữ liệu đào tạo của chúng tôi các đường cong định kỳ có cùng tần số với mùa mà chúng tôi đang cố gắng mô hình hóa. Các đường cong chúng ta sử dụng là các đường cong của các hàm lượng giác sin và cosin.
Fourier features là các cặp đường cong hình sin và cosin, một cặp cho mỗi tần số tiềm năng trong mùa bắt đầu với tần số dài nhất. Các cặp Fourier mô hình hóa tính thời vụ hàng năm sẽ có tần suất: một lần mỗi năm, hai lần mỗi năm, ba lần mỗi năm, v.v.
 ![alt text](https://storage.googleapis.com/kaggle-media/learn/images/bKOjdU7.png  )
<center> 
The first two Fourier pairs for annual seasonality. Top: Frequency of once per year. Bottom: Frequency of twice per year. 
</center>


Nếu chúng ta thêm một tập hợp các đường cong sin / cosin này vào dữ liệu đào tạo của mình, thuật toán hồi quy tuyến tính sẽ tìm ra các trọng số phù hợp với thành phần theo mùa trong chuỗi mục tiêu. Hình vẽ minh họa cách hồi quy tuyến tính sử dụng bốn cặp Fourier để mô hình hóa tính thời vụ hàng năm trong Wiki Trigonometry series.
 

![alt text](https://storage.googleapis.com/kaggle-media/learn/images/mijPhko.png )
<center> 
Trên cùng: Đường cong cho bốn cặp Fourier, tổng sin và cosin với hệ số hồi quy. Mỗi đường cong mô hình hóa một tần số khác nhau. Dưới cùng: Tổng của các đường cong này xấp xỉ mô hình theo mùa.
</center>

Lưu ý rằng chúng tôi chỉ cần tám tính năng (bốn cặp sin / cosin) để có được ước tính tốt về tính thời vụ hàng năm. So sánh điều này với phương pháp chỉ báo theo mùa đòi hỏi hàng trăm tính năng (một tính năng cho mỗi ngày trong năm). Bằng cách chỉ mô hình hóa "hiệu ứng chính" của tính thời vụ với các tính năng của Fourier, bạn thường sẽ cần thêm ít tính năng hơn vào dữ liệu đào tạo của mình, điều đó có nghĩa là giảm thời gian tính toán và ít rủi ro quá tải.
#### Choosing Fourier features with the Periodogram
* Chúng ta thực sự nên bao gồm bao nhiêu cặp Fourier trong bộ tính năng của mình? Chúng ta có thể trả lời câu hỏi này bằng chu kỳ. Chu kỳ (P eriodogram ) cho bạn biết cường độ của các tần số trong một chuỗi thời gian. Cụ thể, giá trị trên trục y của đồ thị là (a ** 2 + b ** 2) / 2, trong đó a  và b là các hệ số của sin và cosin ở tần số đó (như trong  biểu đồ Thành phần Fourier ở trên).  
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/PK6WEe3.png  )
<center> 
Periodogram for the Wiki Trigonometry series.
</center>

Từ trái sang phải, chu kỳ giảm xuống sau hàng quý, bốn lần một năm. Đó là lý do tại sao chúng tôi chọn bốn cặp Fourier để mô hình hóa mùa giải hàng năm. Tần  suất hàng tuần chúng tôi bỏ qua vì nó được mô hình hóa tốt hơn với các chỉ số. 
#### Tính năng Computing Fourier (tùy chọn)
Biết cách tính toán các tính năng Fourier không cần thiết để sử dụng chúng, nhưng nếu nhìn thấy các chi tiết sẽ làm rõ mọi thứ, ô ẩn bên dưới minh họa cách một tập hợp các tính năng Fourier có thể được lấy từ chỉ mục của một chuỗi thời gian. (Chúng ta sẽ sử dụng hàm thư viện từ Tuy nhiên, StatsModels cho các ứng dụng của chúng tôi.)
``` python
import numpy as np


def fourier_features(index, freq, order):
    time = np.arange(len(index), dtype=np.float32)
    k = 2 * np.pi * (1 / freq) * time
    features = {}
    for i in range(1, order + 1):
        features.update({
            f"sin_{freq}_{i}": np.sin(i * k),
            f"cos_{freq}_{i}": np.cos(i * k),
        })
    return pd.DataFrame(features, index=index)

# Compute Fourier features to the 4th order (8 new features) for a
# series y with daily observations and annual seasonality:
#
# fourier_features(y, freq=365.25, order=4)
``` 
## What is Serial Dependence?

* Trong các bài học trước, chúng tôi đã điều tra các thuộc tính của chuỗi thời gian được mô hình hóa dễ dàng nhất là  các thuộc tính phụ thuộc thời gian,  nghĩa là, với các tính năng chúng tôi có thể rút ra trực tiếp từ chỉ số thời gian. Tuy nhiên, một số thuộc tính chuỗi thời gian chỉ có thể được mô hình hóa dưới dạng  các thuộc tính phụ thuộc tuần tự, nghĩa là sử dụng làm tính năng các giá trị trong quá khứ của chuỗi đích. Cấu trúc của các chuỗi thời gian này có thể không rõ ràng từ một cốt truyện theo thời gian; Tuy nhiên, được vẽ dựa trên các giá trị trong quá khứ, cấu trúc trở nên rõ ràng - như chúng ta thấy trong hình dưới đây.
 
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/X0sSnwp.png) 
<center>
Hai chuỗi này có sự phụ thuộc nối tiếp, nhưng không phụ thuộc thời gian. Các điểm bên phải có tọa độ  (giá trị tại thời điểm t-1, giá trị tại thời điểm t).
</center>

With trend and seasonality, Chúng tôi đã đào tạo các mô hình để phù hợp với các đường cong với các ô như các mô hình bên trái trong hình trên - các mô hình đang học sự phụ thuộc thời gian. Mục tiêu trong bài học này là đào tạo các mô hình để phù hợp với các đường cong với các ô như các mô hình bên phải - chúng tôi muốn chúng học cách phụ thuộc nối tiếp.
#### Cycles
* Một cách đặc biệt phổ biến để sự phụ thuộc nối tiếp biểu hiện là theo chu kỳ. Chu kỳ là các mô hình tăng trưởng và phân rã trong một chuỗi thời gian liên quan đến cách giá trị trong một chuỗi tại một thời điểm phụ thuộc vào các giá trị tại các thời điểm trước đó, nhưng không nhất thiết phải dựa vào chính bước thời gian. Hành vi tuần hoàn là đặc trưng của các hệ thống có thể ảnh hưởng đến bản thân hoặc có phản ứng tồn tại theo thời gian. Các nền kinh tế, dịch bệnh, quần thể động vật, núi lửa phun trào và các hiện tượng tự nhiên tương tự thường thể hiện hành vi theo chu kỳ.

![alt text](https://storage.googleapis.com/kaggle-media/learn/images/CC3TkAf.png)
<center>
Bốn chuỗi thời gian với hành vi tuần hoàn.
</center>

Điều phân biệt hành vi tuần hoàn với tính thời vụ là các chu kỳ không nhất thiết phải phụ thuộc vào thời gian, như các mùa. Những gì xảy ra trong một chu kỳ ít liên quan đến ngày xảy ra cụ thể và nhiều hơn về những gì đã xảy ra trong quá khứ gần đây. Sự độc lập (ít nhất là tương đối) theo thời gian có nghĩa là hành vi tuần hoàn có thể bất thường hơn nhiều so với tính thời vụ(seasonality).
#### Lagged Series and Lag Plots
* Để điều tra sự phụ thuộc nối tiếp có thể xảy ra (like cycles) trong một chuỗi thời gian, chúng ta cần tạo "lagged" copies of the series. Lagging a time series có nghĩa là dịch chuyển các giá trị của nó về phía trước một hoặc nhiều bước thời gian, hoặc tương đương, để dịch chuyển thời gian trong chỉ số của nó lùi lại một hoặc nhiều bước. Trong cả hai trường hợp, hiệu ứng là các quan sát trong chuỗi bị trễ sẽ xuất hiện muộn hơn trong thời gian.

Điều này cho thấy tỷ lệ thất nghiệp hàng tháng ở Mỹ (y) cùng với first and second lagged series (y_lag_1 and y_lag_2, respectively). Lưu ý cách các giá trị của chuỗi bị trễ được dịch chuyển về phía trước theo thời gian.
In [1]:
```python
import pandas as pd

# Federal Reserve dataset: https://www.kaggle.com/federalreserve/interest-rates
reserve = pd.read_csv(
    "../input/ts-course-data/reserve.csv",
    parse_dates={'Date': ['Year', 'Month', 'Day']},
    index_col='Date',
)

y = reserve.loc[:, 'Unemployment Rate'].dropna().to_period('M')
df = pd.DataFrame({
    'y': y,
    'y_lag_1': y.shift(1),
    'y_lag_2': y.shift(2),    
})

df.head()
```
Out[1]:
|    Date   |   y   | y_lag_1 | y_lag_2 |
|:---------:|:-----:|:-------:|:-------:|
| 1954-07   |  5.8  |   NaN   |   NaN   |
| 1954-08   |  6.0  |   5.8   |   NaN   |
| 1954-09   |  6.1  |   6.0   |   5.8   |
| 1954-10   |  5.7  |   6.1   |   6.0   |
| 1954-11   |  5.3  |   5.7   |   6.1   |


Bằng cách tụt hậu một chuỗi thời gian, chúng ta có thể làm cho các giá trị trong quá khứ của nó xuất hiện đồng thời với các giá trị mà chúng ta đang cố gắng dự đoán (nói cách khác là trong cùng một hàng). Điều này làm cho chuỗi trễ trở nên hữu ích như các tính năng để mô hình hóa sự phụ thuộc nối tiếp. Để dự báo chuỗi tỷ lệ thất nghiệp của Mỹ, chúng ta có thể sử dụng y_lag_1  và y_lag_2 làm các tính năng để dự đoán mục tiêu y. Điều này sẽ dự báo tỷ lệ thất nghiệp trong tương lai như một chức năng của tỷ lệ thất nghiệp trong hai tháng trước.
#### Lag plots
* A lag plot của một chuỗi thời gian cho thấy các giá trị của nó được vẽ dựa trên độ trễ của nó. Sự phụ thuộc nối tiếp trong một chuỗi thời gian thường sẽ trở nên rõ ràng bằng cách nhìn vào một biểu đồ trễ. Chúng ta có thể thấy từ biểu đồ tụt hậu này của Thất nghiệp Hoa Kỳ rằng có một mối quan hệ tuyến tính mạnh mẽ và rõ ràng giữa tỷ lệ thất nghiệp hiện tại và tỷ lệ trong quá khứ.

![alt text](https://storage.googleapis.com/kaggle-media/learn/images/Hvrboya.png)
<center>
Lag plot of US Unemployment with autocorrelations indicated.
</center> 

Thước đo phụ thuộc nối tiếp được sử dụng phổ biến nhất được gọi  là tự tương quan, đơn giản là mối tương quan mà một chuỗi thời gian có với một trong những độ trễ của nó. Thất nghiệp Hoa Kỳ có mối tương quan tự động là 0,99 ở độ trễ 1, 0,98 ở độ trễ 2, v.v.
#### Choosing lags
* Khi chọn độ trễ để sử dụng làm tính năng, thường sẽ không hữu ích khi bao gồm mọi độ trễ có tương quan tự động lớn.  Ví dụ, trong tình trạng thất nghiệp ở Mỹ, mối tương quan tự động ở độ trễ 2 có thể hoàn toàn là kết quả của thông tin "phân rã" từ độ trễ 1 - chỉ là mối tương quan được chuyển từ bước trước. Nếu lag 2 không chứa bất kỳ thứ gì mới, sẽ không có lý do gì để đưa nó vào nếu chúng ta đã 
* có độ trễ 1.

Tự tương quan một phần cho bạn biết mối tương quan của độ trễ chiếm tất cả các độ trễ trước đó - số lượng tương quan "mới" mà độ trễ đóng góp, có thể nói như vậy. Vẽ biểu đồ tự động tương quan một phần có thể giúp bạn chọn tính năng độ trễ nào sẽ sử dụng. Trong hình dưới đây, độ trễ từ 1 đến độ trễ 6 nằm ngoài khoảng thời gian "không có tương quan" (màu xanh lam), vì vậy chúng ta có thể chọn độ trễ từ 1 đến độ trễ 6 làm tính năng cho Thất nghiệp Hoa Kỳ. (Độ trễ 11 có thể là dương tính giả.)
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/6nTe94E.png )
<center>
Tự động tương quan một phần của Thất nghiệp Hoa Kỳ thông qua độ trễ 12 với khoảng tin cậy 95% không có tương quan. 
</center> 

Một cốt truyện như thế ở trên được gọi là correlogram. Correlogram dành cho các tính năng độ trễ về cơ bản là những gì định kỳ dành cho các tính năng Fourier.

Cuối cùng, chúng ta cần lưu ý rằng tự tương quan và tự tương quan một phần là thước đo của  sự phụ thuộc tuyến tính. Bởi vì chuỗi thời gian trong thế giới thực thường có sự phụ thuộc phi tuyến tính đáng kể, tốt nhất bạn nên xem xét biểu đồ độ trễ (hoặc sử dụng một số thước đo phụ thuộc tổng quát hơn, như mutual information) Một cốt truyện như thế ở trên được gọi là correlogram. Correlogram dành cho các tính năng độ trễ về cơ bản là những gì định kỳ dành cho các tính năng Fourier.

Cuối cùng, chúng ta cần lưu ý rằng tự tương quan và tự tương quan một phần là thước đo của  sự phụ thuộc tuyến tính. Bởi vì chuỗi thời gian trong thế giới thực thường có sự phụ thuộc phi tuyến tính đáng kể, tốt nhất bạn nên xem xét biểu đồ độ trễ (hoặc sử dụng một số thước đo phụ thuộc tổng quát hơn, như .
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/Q38UVOu.png  )
<center>
Cốt truyện trễ của  loạt Sunspot.
</center> 

Các mối quan hệ phi tuyến tính như thế này có thể được chuyển đổi thành tuyến tính hoặc được học bằng một thuật toán thích hợp.

## Hybrid Models
* Hồi quy tuyến tính vượt trội trong việc ngoại suy xu hướng, nhưng không thể học các tương tác. XGBoost vượt trội trong việc học các tương tác, nhưng không thể ngoại suy xu hướng. Trong bài học này, chúng ta sẽ học cách tạo ra các nhà dự báo "lai" kết hợp các thuật toán học tập bổ sung và để điểm mạnh của cái này bù đắp cho điểm yếu của cái kia.
#### Components and Residuals
* Để chúng ta có thể thiết kế các giống lai hiệu quả, chúng ta cần hiểu rõ hơn về cách xây dựng chuỗi thời gian. Cho đến nay, chúng tôi đã nghiên cứu ba mô hình phụ thuộc: trend, seasons, and cycles. Nhiều chuỗi thời gian có thể được mô tả chặt chẽ bằng một mô hình phụ gia chỉ gồm ba thành phần này cộng với một số về cơ bản không thể đoán trước, hoàn toàn ngẫu nhiên error:
**series = trend + seasons + cycles + error**

Mỗi thuật ngữ trong mô hình này sau đó chúng ta sẽ gọi là một component of the time series.

Phần còn lại(residuals) của một mô hình là sự khác biệt giữa mục tiêu mà mô hình  được đào tạo và các dự đoán mà mô hình đưa ra - nói cách khác là sự khác biệt giữa đường cong thực tế và đường cong được trang bị. Vẽ phần còn lại so với một tính năng và bạn sẽ nhận được phần "còn sót lại" của mục tiêu hoặc những gì mô hình không tìm hiểu về mục tiêu từ tính năng đó.
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/mIeeaBD.png )
<center>
Sự khác biệt giữa chuỗi mục tiêu và dự đoán (màu xanh lam) cho chuỗi phần dư.
</center>  

Bên trái hình trên là một phần của  chuỗi Giao thông đường hầm và đường cong theo mùa theo xu hướng từ Bài học 3. Trừ đi đường cong vừa vặn để lại phần dư, bên phải. Phần còn lại chứa mọi thứ từ Giao thông đường hầm, mô hình theo mùa theo xu hướng không học được.

Chúng ta có thể tưởng tượng việc học các thành phần của một chuỗi thời gian như một quá trình lặp đi lặp lại: đầu tiên tìm hiểu xu hướng(trend) và trừ nó ra khỏi chuỗi, sau đó tìm hiểu tính thời vụ (seasonality) từ các phần dư bị giảm xu hướng và trừ đi các mùa, sau đó tìm hiểu các chu kỳ(cycle) và trừ đi các chu kỳ, và cuối cùng chỉ còn lại lỗi không thể đoán trước.
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/XGJuheO.png)
<center>
Tìm hiểu các thành phần của Mauna Loa CO2 từng bước. Trừ đi đường cong được trang bị (màu xanh lam) khỏi chuỗi của nó để có được chuỗi trong bước tiếp theo.
</center>  

Thêm tất cả các thành phần chúng tôi đã học và chúng tôi có được mô hình hoàn chỉnh. Về cơ bản, đây là những gì hồi quy tuyến tính sẽ làm nếu bạn đào tạo nó trên một bộ đầy đủ các tính năng mô hình hóa xu hướng, mùa và chu kỳ.
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/HZEhuHF.png)
<center>
Thêm các thành phần đã học để có được một mô hình hoàn chỉnh. 
</center>  


#### Hybrid Forecasting with Residuals
* Trong các bài học trước, chúng tôi đã sử dụng một thuật toán duy nhất (hồi quy tuyến tính) để tìm hiểu tất cả các thành phần cùng một lúc. Nhưng cũng có thể sử dụng một thuật toán cho một số thành phần và một thuật toán khác cho phần còn lại. Bằng cách này, chúng tôi luôn có thể chọn thuật toán tốt nhất cho từng thành phần. Để làm điều này, chúng tôi sử dụng một thuật toán để phù hợp với chuỗi gốc và sau đó là thuật toán thứ hai để phù hợp với chuỗi dư.
```python
In detail, the process is this:
# 1. Train and predict with first model
model_1.fit(X_train_1, y_train)
y_pred_1 = model_1.predict(X_train)

# 2. Train and predict with second model on residuals
model_2.fit(X_train_2, y_train - y_pred_1)
y_pred_2 = model_2.predict(X_train_2)

# 3. Add to get overall predictions
y_pred = y_pred_1 + y_pred_2
```
Chúng tôi thường sẽ muốn sử dụng các bộ tính năng khác nhau (X_train_1 và X_train_2 ở trên) tùy thuộc vào những gì chúng tôi muốn mỗi mô hình tìm hiểu. Ví dụ: nếu chúng ta sử dụng mô hình đầu tiên để tìm hiểu xu hướng, chúng ta thường sẽ không cần tính năng xu hướng cho mô hình thứ hai.

Mặc dù có thể sử dụng nhiều hơn hai mô hình, nhưng trong thực tế, nó dường như không đặc biệt hữu ích. Trên thực tế, chiến lược phổ biến nhất để xây dựng các giống lai là chiến lược mà chúng tôi vừa mô tả: một thuật toán học tập đơn giản (thường là tuyến tính) theo sau là một người học phi tuyến tính, phức tạp như GBDT hoặc mạng lưới thần kinh sâu, mô hình đơn giản thường được thiết kế như một "người trợ giúp" cho thuật toán mạnh mẽ theo sau.
#### Designing Hybrids
* Có nhiều cách bạn có thể kết hợp các mô hình học máy bên cạnh cách chúng tôi đã nêu trong bài học này. Tuy nhiên, việc kết hợp thành công các mô hình đòi hỏi chúng ta phải đào sâu hơn một chút vào cách các thuật toán này hoạt động.

Nói chung có hai cách mà một thuật toán hồi quy có thể đưa ra dự đoán: bằng  cách chuyển đổi các tính năng hoặc bằng cách chuyển đổi mục tiêu. Các thuật toán chuyển đổi tính năng tìm hiểu một số hàm toán học lấy các tính năng làm đầu vào và sau đó kết hợp và biến đổi chúng để tạo ra đầu ra khớp với các giá trị đích trong tập đào tạo. Linear regression and neural nets đều thuộc loại này.

Các thuật toán chuyển đổi mục tiêu sử dụng các tính năng để nhóm các giá trị mục tiêu trong bộ đào tạo và đưa ra dự đoán bằng cách tính trung bình các giá trị trong một nhóm; Một tập hợp các tính năng chỉ cho biết nhóm nào cần tính trung bình. Decision trees and nearest neighbors đều thuộc loại này.

Điều quan trọng là điều này: feature transformers Nói chung có thể ngoại suy các giá trị mục tiêu ngoài bộ đào tạo  được cung cấp các tính năng thích hợp làm đầu vào, nhưng dự đoán của máy biến áp mục tiêu sẽ luôn bị ràng buộc trong phạm vi của bộ đào tạo. Nếu hình nộm thời gian tiếp tục đếm các bước thời gian, hồi quy tuyến tính tiếp tục vẽ đường xu hướng. Với cùng một hình nộm thời gian, một cây quyết định sẽ dự đoán xu hướng được chỉ ra bởi bước cuối cùng của dữ liệu đào tạo trong tương lai mãi mãi. Cây quyết định không thể ngoại suy xu hướng. Rừng ngẫu nhiên và cây quyết định tăng cường độ dốc (như XGBoost) là tập hợp các cây quyết định, vì vậy chúng cũng không thể ngoại suy xu hướng.
 ![alt text](https://storage.googleapis.com/kaggle-media/learn/images/ZZtfuFJ.png)
<center>
Một cây quyết định sẽ không ngoại suy một xu hướng ngoài bộ đào tạo. 
</center>  

Sự khác biệt này là điều thúc đẩy thiết kế lai trong bài học này: sử dụng hồi quy tuyến tính để ngoại suy xu hướng, chuyển đổi mục tiêu để loại bỏ xu hướng và áp dụng XGBoost cho phần dư đã mất xu hướng. Để lai một mạng lưới thần kinh (A feature transformer), thay vào đó bạn có thể bao gồm các dự đoán của một mô hình khác như một tính năng, mà mạng lưới thần kinh sau đó sẽ bao gồm như một phần của dự đoán của riêng nó. Phương pháp phù hợp với phần dư thực sự là cùng một phương pháp mà thuật toán gradient boosting sử dụng, vì vậy chúng tôi sẽ gọi chúng là các giống lai được tăng cường; phương pháp sử dụng dự đoán làm tính năng được gọi là "xếp chồng", vì vậy chúng tôi sẽ gọi chúng  là  lai xếp chồng lên nhaus.
## Forecasting With Machine Learning
* Giới thiệu
Trong Bài học 2 và 3, chúng tôi coi dự báo là một vấn đề hồi quy đơn giản với tất cả các tính năng của chúng tôi bắt nguồn từ một đầu vào duy nhất, chỉ số thời gian. Chúng tôi có thể dễ dàng tạo dự báo cho bất kỳ thời điểm nào trong tương lai bằng cách tạo ra xu hướng mong muốn và các tính năng theo mùa.
Tuy nhiên, khi chúng tôi thêm các tính năng độ trễ trong Bài học 4, bản chất của vấn đề đã thay đổi. Các tính năng độ trễ yêu cầu giá trị mục tiêu bị trễ phải được biết tại thời điểm được dự báo. Tính năng lag 1 chuyển chuỗi thời gian về phía trước 1 bước, có nghĩa là bạn có thể dự đoán 1 bước trong tương lai nhưng không phải 2 bước.
Trong Bài học 4, chúng tôi chỉ giả định rằng chúng tôi luôn có thể tạo ra độ trễ cho đến khoảng thời gian chúng tôi muốn dự báo (nói cách khác, mọi dự đoán chỉ là một bước tiến). Dự báo trong thế giới thực thường đòi hỏi nhiều hơn thế này, vì vậy trong bài học này, chúng ta sẽ học cách đưa ra dự báo cho nhiều tình huống khác nhau.
#### Defining the Forecasting Task
* Có hai điều cần thiết lập trước khi thiết kế một mô hình dự báo:
    1.	thông tin nào có sẵn tại thời điểm dự báo được thực hiện (tính năng) và,
    2.	Khoảng thời gian mà bạn yêu cầu giá trị dự báo (mục tiêu).

Nguồn gốc dự báo  là thời gian bạn đưa ra dự báo. Trên thực tế, bạn có thể coi nguồn gốc dự báo là lần cuối cùng bạn có dữ liệu đào tạo trong thời gian được dự đoán. Mọi thứ cho đến nguồn gốc của anh ta có thể được sử dụng để tạo ra các tính năng.
Đường chân trời dự báo  là thời gian mà bạn đang đưa ra dự báo. Chúng ta thường mô tả một dự báo theo số bước thời gian trong đường chân trời của nó: dự báo "1 bước" hoặc dự báo "5 bước". Chân trời dự báo mô tả mục tiêu.
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/xwEgcOk.png )
<center>
Chân trời dự báo ba bước với thời gian thực hiện hai bước, sử dụng bốn tính năng độ trễ. Con số này đại diện cho những gì sẽ là một hàng dữ liệu đào tạo duy nhất - dữ liệu cho một dự đoán duy nhất, nói cách khác.
</center>  

Thời gian giữa điểm xuất phát và đường chân trời là thời gian dẫn (hoặc đôi khi độ trễ) của dự báo. Thời gian thực hiện của dự báo được mô tả bằng số bước từ điểm xuất phát đến đường chân trời: dự báo "đi trước 1 bước" hoặc "đi trước 3 bước". Trong thực tế, dự báo có thể cần phải bắt đầu nhiều bước trước nguồn gốc vì sự chậm trễ trong việc thu thập hoặc xử lý dữ liệu.
####P reparing Data for Forecasting
* Để dự báo chuỗi thời gian với các thuật toán ML, chúng ta cần chuyển đổi chuỗi thành một khung dữ liệu mà chúng ta có thể sử dụng với các thuật toán đó. (Tất nhiên, trừ khi bạn chỉ sử dụng các tính năng xác định như trend and seasonality.)
Chúng ta đã thấy nửa đầu của quá trình này trong Bài học 4 khi chúng ta tạo ra một tính năng được thiết lập từ độ trễ. Hiệp hai đang chuẩn bị mục tiêu. Làm thế nào chúng ta làm điều này phụ thuộc vào nhiệm vụ dự báo.

Mỗi hàng trong khung dữ liệu đại diện cho một dự báo duy nhất. Chỉ số thời gian của hàng là lần đầu tiên trong đường chân trời dự báo, nhưng chúng tôi sắp xếp các giá trị cho toàn bộ đường chân trời trong cùng một hàng. Đối với dự báo nhiều bước, điều này có nghĩa là chúng tôi đang yêu cầu một mô hình tạo ra nhiều đầu ra, một cho mỗi bước.

In [1]:
```python
import numpy as np
import pandas as pd

N = 20
ts = pd.Series(
    np.arange(N),
    index=pd.period_range(start='2010', freq='A', periods=N, name='Year'),
    dtype=pd.Int8Dtype,
)

# Lag features
X = pd.DataFrame({
    'y_lag_2': ts.shift(2),
    'y_lag_3': ts.shift(3),
    'y_lag_4': ts.shift(4),
    'y_lag_5': ts.shift(5),
    'y_lag_6': ts.shift(6),    
})

# Multistep targets
y = pd.DataFrame({
    'y_step_3': ts.shift(-2),
    'y_step_2': ts.shift(-1),
    'y_step_1': ts,
})

data = pd.concat({'Targets': y, 'Features': X}, axis=1)

data.head(10).style.set_properties(['Targets'], **{'background-color': 'LavenderBlush'}) \
                   .set_properties(['Features'], **{'background-color': 'Lavender'})
```
Out[1]:
                    
 |  Year  | y_step_3 | y_step_2 | y_step_1 | y_lag_2 | y_lag_3 | y_lag_4 | y_lag_5 | y_lag_6 |
|:------:|:--------:|:--------:|:--------:|:-------:|:-------:|:-------:|:-------:|:-------:|
|  2010  |    2     |    1     |    0     |   nan   |   nan   |   nan   |   nan   |   nan   |
|  2011  |    3     |    2     |    1     |   nan   |   nan   |   nan   |   nan   |   nan   |
|  2012  |    4     |    3     |    2     |    0    |   nan   |   nan   |   nan   |   nan   |
|  2013  |    5     |    4     |    3     |    1    |    0    |   nan   |   nan   |   nan   |
|  2014  |    6     |    5     |    4     |    2    |    1    |    0    |   nan   |   nan   |
|  2015  |    7     |    6     |    5     |    3    |    2    |    1    |    0    |   nan   |
|  2016  |    8     |    7     |    6     |    4    |    3    |    2    |    1    |    0    |
|  2017  |    9     |    8     |    7     |    5    |    4    |    3    |    2    |    1    |
|  2018  |   10     |    9     |    8     |    6    |    5    |    4    |    3    |    2    |
|  2019  |   11     |   10     |    9     |    7    |    6    |    5    |    4    |    3    |

Ở trên minh họa cách một tập dữ liệu sẽ được chuẩn bị tương tự như  Xác định số liệu Dự báo: một nhiệm vụ dự báo ba bước với thời gian thực hiện hai bước bằng cách sử dụng năm tính năng độ trễ. Chuỗi thời gian gốc là y_step_1. Các giá trị còn thiếu chúng ta có thể điền hoặc thả.
#### Multistep Forecasting Strategies
* Có một số chiến lược để tạo ra nhiều bước mục tiêu cần thiết cho một dự báo. Chúng tôi sẽ phác thảo bốn chiến lược chung, mỗi chiến lược đều có điểm mạnh và 
điểm yếu.

1. Multioutput model
* Sử dụng một mô hình tạo ra nhiều đầu ra một cách tự nhiên. Hồi quy tuyến tính và mạng lưới thần kinh đều có thể tạo ra nhiều đầu ra. Chiến lược này đơn giản và hiệu quả, nhưng không thể thực hiện được đối với mọi thuật toán bạn có thể muốn sử dụng. Ví dụ, XGBoost không thể làm điều này.
 
![alt text](https://storage.googleapis.com/kaggle-media/learn/images/uFsHiqr.png) 

2. Direct strategy
* Đào tạo một mô hình riêng biệt cho mỗi bước trong đường chân trời: một mô hình dự báo trước 1 bước, một mô hình khác đi trước 2 bước, v.v. Dự báo trước 1 bước là một vấn đề khác với 2 bước trước (v.v.), vì vậy có thể giúp có một mô hình khác nhau đưa ra dự báo cho từng bước. Nhược điểm là việc đào tạo nhiều mô hình có thể tốn kém về mặt tính toán.
 ![alt text](https://storage.googleapis.com/kaggle-media/learn/images/HkolNMV.png ) 


3. Recursive strategy
* Đào tạo một mô hình một bước duy nhất và sử dụng dự báo của mô hình đó để cập nhật các tính năng độ trễ cho bước tiếp theo. Với phương pháp đệ quy, chúng tôi đưa dự báo 1 bước của mô hình trở lại cùng mô hình đó để sử dụng làm tính năng độ trễ cho bước dự báo tiếp theo. Chúng ta chỉ cần đào tạo một mô hình, nhưng vì các lỗi sẽ lan truyền từ bước này sang bước khác, dự báo có thể không chính xác cho những chân trời dài.
 ![alt text](https://storage.googleapis.com/kaggle-media/learn/images/sqkSFDn.png)
4. DirRec strategy
* Sự kết hợp của các chiến lược trực tiếp và đệ quy: đào tạo mô hình cho từng bước và sử dụng dự báo từ các bước trước làm  tính năng độ trễ mới. Từng bước, mỗi mô hình nhận được một đầu vào độ trễ bổ sung. Vì mỗi mô hình luôn có một bộ tính năng độ trễ cập nhật, chiến lược DirRec có thể nắm bắt sự phụ thuộc nối tiếp tốt hơn Direct, nhưng nó cũng có thể bị lan truyền lỗi như đệ quy.
 ![alt text](https://storage.googleapis.com/kaggle-media/learn/images/B7KAvAO.png )






