// JavaScript Document
$(document).ready(function(e) {

	linum = $('.mainlist a').length;//图片数量
	console.log(linum)
	w = linum * 135;//ul宽度
	// $('./**/piclist').css('width', w + 'px');//ul宽度
	// $('.swaplist').html($('.mainlist').html());//复制内容
	
	$('.og_next').click(function(){


		
		if($('.mainlist li').length>4){//多于4张图片
			ml = parseInt($('.mainlist').css('left'));//默认图片ul位置
			// sl = parseInt($('.swaplist').css('left'));//交换图片ul位置
			console.log("you")
			console.log(w);
			console.log(ml);
			// console.log(sl);
			if(ml<510 && ml>w*-1){//默认图片显示时
				// $('.swaplist').css({left: '510px'});//交换图片放在显示区域右侧
				// if(w+ml>510){
				  $('.mainlist').animate({left: ml - 510 + 'px'},'slow');//默认图片滚动

                // }
			}
		}
	})
	$('.og_prev').click(function(){

		
		if($('.mainlist a').length>4){
			ml = parseInt($('.mainlist').css('left'));
			console.log("zuo")
			console.log(w);
			console.log(ml);
			// console.log(sl);
			// sl = parseInt($('.swaplist').css('left'));
			if(ml>(510+w)*-1 && ml<0){

				$('.mainlist').animate({left: ml + 510 + 'px'},'slow');

			}
		}
	})    
});

$(document).ready(function(){
	$('.og_prev,.og_next').hover(function(){
			$(this).fadeTo('fast',1);
		},function(){
			$(this).fadeTo('fast',0.7);
	})

})

