/**
 * Ataa Slider.
 * by NetMechanics
 * Eng\ Mahmoud Alghool
 * www.netmechanics.net
 * info@netmechanics.net
 */

ataaSlider = null;
imgsArr = [];
structure = '<div id="slider-holder">' +
              '<a id="moveRight"></a>' +
              '<a id="openImage"></a>' +
              '<a id="moveLeft"></a>' +
              '<div id="images-holder">' +
                '<div class="slide "></div>' +
                '<div class="slide "></div>' +
                '<div class="slide "></div>' +
                '<div class="slide "></div>' +
                '<div class="slide "></div>' +
              '</div>' +
              '<div id="control-holder">' +
                '<div class="left-holder"><i class="first left glyphicon glyphicon-menu-left"></i><i class="second left glyphicon glyphicon-menu-left"></i></div>' +
                '<div class="main-holder"><p style="display: none"></p></div>' +
                '<div class="right-holder"><i class="first right glyphicon glyphicon-menu-right"></i><i class="second right glyphicon glyphicon-menu-right"></i></div>' +
              '</div>' +
            '</div>';
imgStructure = '<div class="img-holder"><span class="img-name"></span><img src="" data-key="" alt=""></div>';
imgShowStructure = '<div id="image-show" style="display: none"><span class="image-close"><i class="glyphicon glyphicon-remove" ></i></span><img src="" alt=""></div>';
options = {};
moveInterval = 0;
acting = true;

$.fn.ataaSlider = function(userOptions){
  options = $.extend({}, $.fn.ataaSlider.defaults, userOptions);
  ataaSlider = $(this);
  initialize();
  $('#moveRight').on('click',function(){
    moveWrappper(-1);
  });
  $('#moveLeft').on('click',function(){
    moveWrappper(1);
  });
  if(options.openImage){
    $('#openImage').on('click', function(){
      if(acting) {
        console.log("inacting");
        return;
      }
      acting = true;
      var key = $(ataaSlider.find('.slide')[2]).find('img').data('key');
      openimage(imgsArr[key]);
      $('#image-show').on('click',function(){

        $('body #image-show').fadeOut(function(){
          $('body #image-show').remove();
          acting = false;
        });

      });
    });

  }else{
    $('#openImage').remove();
  }

  window.onresize = function() {
    sizing();
  }
};

$.fn.ataaSlider.defaults =
{
  shownames: true,
  height: '400',
  controlheight: '100',
  speed:500,
  openImage: true
};


function initialize(){
  var imgs = ataaSlider.find('div');

  if(imgs.length < 1){
    console.log("ERROR: no images in slider");
    return false;
  }

  //get images info into array
  imgs.each(function(index){
    var thisImage = $(this);
    imgsArr.push({
      key:index,
      src:thisImage.data('src'),
      name:thisImage.data('name'),
      desc:thisImage.data('desc')
    });
  });
  imgs.remove();

  //set images counter
  var i = 0;
  while(imgsArr.length < 5){
    var thisImage = imgsArr[i];
    imgsArr.push({
      key:imgsArr.length,
      src:thisImage.src,
      name:thisImage.name,
      desc:thisImage.desc
    });
    i++;
  }
  ataaSlider.html(structure);
  sizing();

  $('.main-holder p').text(imgsArr[2].desc);
  $('.main-holder p').fadeIn();

  startArrowAnimations();
  acting = false;
}

function sizing(){
  //get positions
  var sliderWidth = ataaSlider.width();
  var width60 = sliderWidth * (60/100);
  var width20 = sliderWidth * (20/100);
  moveInterval = width60 + 10;
  startPosition = - moveInterval -(width60 -(width20 - 10));

  ataaSlider.find('#images-holder').css('height', options.height);
  ataaSlider.find('#control-holder').css('height', options.controlheight);

  var slidePosition = startPosition;
  ataaSlider.find('.slide').each(function(i){
    $(this).html(setImage(imgsArr[i]));
    $(this).css('left', slidePosition);
    if(i ==2) $(this).css('padding-top', '0px');
    slidePosition += moveInterval;
  });
}

function moveWrappper(direction){
  if(acting) {
    console.log("inacting");
    return;
  }
  acting = true;
  $('.main-holder p').fadeOut();
  move(direction, function(key){
    $('.main-holder p').text(imgsArr[key].desc);
    $('.main-holder p').fadeIn();
    acting = false;
  })

}

function move(direction, cb){

  var ready = 0;
  var activeSlider = $(ataaSlider.find('.slide')[2]);
  var activeImg = activeSlider.find('img').data('key');
  var interval = direction * moveInterval;
  var actveKey = 0;
  ataaSlider.find('.slide').each(function(i){
    var left =parseInt($(this).position().left + interval) ;
     if((direction == -1 && i == 3) || (direction == 1 && i == 1)){
      padding = '0px';
         actveKey = $(this).find('img').data('key');
     }else {
      padding = '50px';
    }
    $(this).animate({
      left: left,
      'padding-top':padding,
    },options.speed,function(){
        ready++;
    });
  })

  var whenReady = setInterval(function(){
    if(ready > 3){
      clearInterval(whenReady);
      if(direction == -1){
        ataaSlider.find('.slide').first().remove();
        var nextKey = activeImg + 3;
        if(nextKey >= imgsArr.length ) nextKey -= imgsArr.length;
        var newLeft = parseInt(ataaSlider.find('.slide').last().position().left + moveInterval);
        ataaSlider.find('#images-holder').append('<div class="slide" style="left:'+newLeft+'px ;">' +setImage(imgsArr[nextKey]).prop('outerHTML') + '</div>');

      }else{
        ataaSlider.find('.slide').last().remove();
        var nextKey = activeImg - 3;
        if(nextKey < 0) nextKey += imgsArr.length;
        ataaSlider.find('#images-holder').prepend('<div class="slide" style="left:'+startPosition+'px ;">' +setImage(imgsArr[nextKey]).prop('outerHTML') + '</div>');
      }
      ready = 0;
      cb(actveKey);
    }
  },100)
}

function setImage(img){
  var imgHTML = $(imgStructure);
  imgHTML.find('img').attr('src', img.src);
  imgHTML.find('img').attr('data-key', img.key);
  imgHTML.find('img').attr('data-alt', img.name);
  if(options.shownames){
    imgHTML.find('.img-name').text(img.name);
  }else{
    imgHTML.find('.img-name').remove();
  }

  return imgHTML ;
}

function startArrowAnimations(){
  var AnimationSpeed = options.speed * 3;
  $('.first.right').animate({
    left: '50%',
    opacity: 1
  },AnimationSpeed,function(){
    $('.first.right').css({left: '0', opacity:0})
  });
  $('.second.right').animate({
    left: '100%',
    opacity: 0
  },AnimationSpeed,function(){
    $('.second.right').css({left: '50%', opacity:1})
  });
  $('.first.left').animate({
    right: '50%',
    opacity: 1
  },AnimationSpeed,function(){
    $('.first.left').css({right: '0', opacity:0})
  });
  $('.second.left').animate({
    right: '100%',
    opacity: 0
  },AnimationSpeed,function(){
    $('.second.left').css({right: '50%', opacity:1});
    startArrowAnimations();
  });

}

function openimage(img){
  var imgHTML = $(imgShowStructure);
  imgHTML.find('img').attr('src', img['src']);
  $('body').append(imgHTML);
  $('#image-show').fadeIn();
};

$('#ataa-slider').ataaSlider();