## Additional Information
The dataset was collected with help of students. Twelve students were divided to three groups and each student had iTAG device. They walked inside their limited area with activated iTAG. In long corridor, 18.35m x 3m, we denoted 3 areas which illustrate building's entry: inside, in vestibule and outside. Two smartphones, Sony Xperia XA1, received signals. They located at the start and end of 'in vestibule' area, which has length of 2.35m. Collection of RSSIs lasted for 20 minutes.
There are two datasets: filtered_rssi and raw_rssi. We used feedback filter to smooth RSSI. Raw RSSIs are actual RSSIs that smartphone got.

---

## Variable Information
+ name - MAC address of iTAG
+ locationStatus - one of three possible iTAG location: INSIDE, IN_VESTIBULE, OUTSIDE
+ timestamp - timestamp in milliseconds
+ rssiOne - RSSI on first smartphone
+ rssiTwo - RSSI on second smartphone

---
## refrence

https://archive.ics.uci.edu/dataset/586/ble%2Brssi%2Bdataset%2Bfor%2Bindoor%2Blocalization?utm_source=chatgpt.com
