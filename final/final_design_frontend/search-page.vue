
<template>
  <div>
    <div>Search Results:</div>
    <div>{{inputText}}</div>
    <div>{{inputText_dc}}</div>
    <div class="image-container">

    <img :src="pie_danmuku" alt="请" class="left-img">
    <img :src="pie_comment" alt="等" class="right-img">
    </div>
    <div>{{inputText_cloud}}</div>
    <img :src="imgsrc" alt="待" class="down-img">
    <!-- 展示搜索结果 -->
  </div>
</template>
<style>
.image-container{
  display: flex;
}
.left-img{
  margin-left: 10px;
}
.right-img{
  margin-right: 10px;
}
.down-img{
  display: flex;
}
</style>
<script>
import axios from 'axios';

export default {
  data() {
    return {
      inputText: '',
      imgsrc : '/img/placeholder.png',
      pie_danmuku :'/img/placeholder.png',
      pie_comment :'/img/placeholder.png',
      inputText_dc:'请稍等，计算中',
      inputText_cloud:''
    }
  },
  async created() {
    try {
      // 获取URL中的参数
      // const domain = window.location.hostname;
      const params = new URLSearchParams(window.location.search);
      const param1 = params.get('q');
      const response = await axios.get(`/search?param1=`+param1);
      let inputText_d = response.data.lend;
      let inputText_c = response.data.lenc;
      this.inputText = '共爬取弹幕' + inputText_d +'条，评论'+inputText_c+'条'
      this.inputText_dc = '以下是弹幕与评论中的情感占比'
      this.inputText_cloud = '以下是根据弹幕与评论给出的词云'
      this.imgsrc = response.data.path;
      this.pie_comment = response.data.path_per_c;
      this.pie_danmuku = response.data.path_per_d;

    } catch (error) {
      console.log(error);
    }
  }


}
</script>
