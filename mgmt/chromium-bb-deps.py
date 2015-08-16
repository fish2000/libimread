vars = {
    'eyes-free':
         'http://eyes-free.googlecode.com/svn',
    'webkit_rev':
         '@80fea8b676081ee15aa3507bf68dbfb02857dacf',
    'blink':
         'http://src.chromium.org/blink',
    'skia':
         'http://skia.googlecode.com/svn',
    'google-breakpad':
         'http://google-breakpad.googlecode.com/svn',
    'sawbuck':
         'http://sawbuck.googlecode.com/svn',
    'mozc':
         'http://mozc.googlecode.com/svn',
    'git.chromium.org':
         'https://chromium.googlesource.com',
    'v8-i18n':
         'http://v8-i18n.googlecode.com/svn',
    'selenium':
         'http://selenium.googlecode.com/svn',
    'buildspec_platforms':
         'all',
    'webkit_url':
         'https://chromium.googlesource.com/chromium/blink.git',
    'snappy':
         'http://snappy.googlecode.com/svn',
    'ppapi':
         'http://ppapi.googlecode.com/svn',
    'pywebsocket':
         'http://pywebsocket.googlecode.com/svn',
    'libaddressinput':
         'http://libaddressinput.googlecode.com/svn',
    'pyftpdlib':
         'http://pyftpdlib.googlecode.com/svn',
    'google-url':
         'http://google-url.googlecode.com/svn',
    'googletest':
         'http://googletest.googlecode.com/svn',
    'gyp':
         'http://gyp.googlecode.com/svn',
    'seccompsandbox':
         'http://seccompsandbox.googlecode.com/svn',
    'ots':
         'http://ots.googlecode.com/svn',
    'angleproject':
         'http://angleproject.googlecode.com/svn',
    'pefile':
         'http://pefile.googlecode.com/svn',
    'open-vcdiff':
         'http://open-vcdiff.googlecode.com/svn',
    'linux-syscall-support':
         'http://linux-syscall-support.googlecode.com/svn',
    'jsoncpp':
         'http://svn.code.sf.net/p/jsoncpp/code',
    'webrtc':
         'http://webrtc.googlecode.com/svn',
    'web-page-replay':
         'http://web-page-replay.googlecode.com/svn',
    'libjingle':
         'http://libjingle.googlecode.com/svn',
    'cld2':
         'https://cld2.googlecode.com/svn',
    'google-cache-invalidation-api':
         'http://google-cache-invalidation-api.googlecode.com/svn',
    'jsr-305':
         'http://jsr-305.googlecode.com/svn',
    'angle_revision':
         '6df9b37d8e3aed3aea12058900b7932f911a152a',
    'bidichecker':
         'http://bidichecker.googlecode.com/svn',
    'git_url':
         'https://chromium.googlesource.com',
    'native_client':
         'http://src.chromium.org/native_client',
    'trace-viewer':
         'http://trace-viewer.googlecode.com/svn',
    'leveldb':
         'http://leveldb.googlecode.com/svn',
    'webkit_trunk':
         'http://src.chromium.org/blink/trunk',
    'googlemock':
         'http://googlemock.googlecode.com/svn',
    'grit-i18n':
         'http://grit-i18n.googlecode.com/svn',
    'pdfsqueeze':
         'http://pdfsqueeze.googlecode.com/svn',
    'protobuf':
         'http://protobuf.googlecode.com/svn',
    'smhasher':
         'http://smhasher.googlecode.com/svn',
    'google-toolbox-for-mac':
         'http://google-toolbox-for-mac.googlecode.com/svn',
    'libyuv':
         'http://libyuv.googlecode.com/svn',
    'rlz':
         'http://rlz.googlecode.com/svn',
    'v8':
         'http://v8.googlecode.com/svn',
    'octane-benchmark':
         'http://octane-benchmark.googlecode.com/svn',
    'sfntly':
         'http://sfntly.googlecode.com/svn',
    'sctp-refimpl':
         'https://sctp-refimpl.googlecode.com/svn',
    'libphonenumber':
         'http://libphonenumber.googlecode.com/svn',
    'pymox':
         'http://pymox.googlecode.com/svn',
    'google-safe-browsing':
         'http://google-safe-browsing.googlecode.com/svn'
}

deps = {
    'build':
        Var('git_url') + '/chromium/tools/build.git@d2efd4ddb639ee19a02c3df5c04085250e7e0b81',
    'build/scripts/command_wrapper/bin':
        Var('git_url') + '/chromium/tools/command_wrapper/bin.git@2eeebba9a512cae9e4e9312f5ec728dbdad80bd0',
    'build/scripts/gsd_generate_index':
        Var('git_url') + '/chromium/tools/gsd_generate_index.git@d2f5d5a5d212d8fb337d751c0351644a6ac83ac8',
    'build/scripts/private/data/reliability':
        Var('git_url') + '/chromium/src/chrome/test/data/reliability.git@ba644102a2f81bb33582e9474a10812fef825389',
    'build/scripts/tools/deps2git':
        Var('git_url') + '/chromium/tools/deps2git.git@27ce444f50f3c6732982d225d3f4cf67f0979a98',
    'build/third_party/lighttpd':
        Var('git_url') + '/chromium/deps/lighttpd.git@9dfa55d15937a688a92cbf2b7a8621b0927d06eb',
    'depot_tools':
        Var('git_url') + '/chromium/tools/depot_tools.git@5b23e871ba4c195e0e068efaf1bb7e96f8d8f07b',
    'src/breakpad/src':
        Var('git_url') + '/external/google-breakpad/src.git@36f5a1b72e782414151d68f2467ad47c8d8be96b',
    'src/buildtools':
        Var('git_url') + '/chromium/buildtools.git@93b3d0af1b30db55ee42bd2e983f7753153217db',
    'src/chrome/test/data/perf/canvas_bench':
        Var('git_url') + '/chromium/canvas_bench.git@a7b40ea5ae0239517d78845a5fc9b12976bfc732',
    'src/chrome/test/data/perf/frame_rate/content':
        Var('git_url') + '/chromium/frame_rate/content.git@c10272c88463efeef6bb19c9ec07c42bc8fe22b9',
    'src/media/cdm/ppapi/api':
        Var('git_url') + '/chromium/cdm.git@7b7c6cc620e13c8057b4b6bff19e5955feb2c8fa',
    'src/native_client':
        Var('git_url') + '/native_client/src/native_client.git@4f204a442b5ca08ab5bc49a8f55dbf8859d5d661',
    'src/sdch/open-vcdiff':
        Var('git_url') + '/external/open-vcdiff.git@438f2a5be6d809bc21611a94cd37bfc8c28ceb33',
    'src/testing/gmock':
        Var('git_url') + '/external/googlemock.git@29763965ab52f24565299976b936d1265cb6a271',
    'src/testing/gtest':
        Var('git_url') + '/external/googletest.git@be1868139ffe0ccd0e8e3b37292b84c821d9c8ad',
    'src/third_party/WebKit':
        Var('webkit_url') + '' + Var('webkit_rev'),
    'src/third_party/angle':
        Var('git_url') + '/angle/angle.git' + '@' + Var('angle_revision'),
    'src/third_party/bidichecker':
        Var('git_url') + '/external/bidichecker/lib.git@97f2aa645b74c28c57eca56992235c79850fa9e0',
    'src/third_party/boringssl/src':
        'https://boringssl.googlesource.com/boringssl.git@b8cbbec76bb6e8c0b7cbd20bba06a1516ef26c23',
    'src/third_party/brotli/src':
        Var('git_url') + '/external/font-compression-reference.git@8c9c83426beb4a58da34be76ea1fccb4054c4703',
    'src/third_party/cacheinvalidation/src':
        Var('git_url') + '/external/google-cache-invalidation-api/src.git@0fbfe801cca467fa986ebe08d34012342aa47e55',
    'src/third_party/cld_2/src':
        Var('git_url') + '/external/cld2.git@14d9ef8d4766326f8aa7de54402d1b9c782d4481',
    'src/third_party/colorama/src':
        Var('git_url') + '/external/colorama.git@799604a1041e9b3bc5d2789ecbd7e8db2e18e6b8',
    'src/third_party/crashpad/crashpad':
        Var('git_url') + '/crashpad/crashpad.git@5d0a133ecd6625f30c6491d502f12f26a0bef32d',
    'src/third_party/ffmpeg':
        Var('git_url') + '/chromium/third_party/ffmpeg.git@d4b1674dcd2f742403179a3ef8e6dd8d7aaecf1a',
    'src/third_party/flac':
        Var('git_url') + '/chromium/deps/flac.git@0635a091379d9677f1ddde5f2eec85d0f096f219',
    'src/third_party/hunspell':
        Var('git_url') + '/chromium/deps/hunspell.git@c956c0e97af00ef789afb2f64d02c9a5a50e6eb1',
    'src/third_party/hunspell_dictionaries':
        Var('git_url') + '/chromium/deps/hunspell_dictionaries.git@94eb94ce85661b54bb3c27bedeff5428b4a2798e',
    'src/third_party/icu':
        Var('git_url') + '/chromium/deps/icu.git@305d288905bf6c13c4dee12576f36e7ace7311ce',
    'src/third_party/jsoncpp/source/include':
        Var('git_url') + '/external/jsoncpp/jsoncpp/include.git@b0dd48e02b6e6248328db78a65b5c601f150c349',
    'src/third_party/jsoncpp/source/src/lib_json':
        Var('git_url') + '/external/jsoncpp/jsoncpp/src/lib_json.git@a8caa51ba2f80971a45880425bf2ae864a786784',
    'src/third_party/leveldatabase/src':
        Var('git_url') + '/external/leveldb.git@251ebf5dc70129ad3c38193fe6c99a5b0ec6b9fa',
    'src/third_party/libaddressinput/src':
        Var('git_url') + '/external/libaddressinput.git@61f63da7ae6fa469138d60dec5d6bbecc6ab43d6',
    'src/third_party/libexif/sources':
        Var('git_url') + '/chromium/deps/libexif/sources.git@ed98343daabd7b4497f97fda972e132e6877c48a',
    'src/third_party/libjingle/source/talk':
        Var('git_url') + '/external/webrtc/trunk/talk.git@ff9171e7ac5309d1e5863c073315e29ddedbeb6a',
    'src/third_party/libjpeg_turbo':
        Var('git_url') + '/chromium/deps/libjpeg_turbo.git@034e9a9747e0983bc19808ea70e469bc8342081f',
    'src/third_party/libphonenumber/src/phonenumbers':
        Var('git_url') + '/external/libphonenumber/cpp/src/phonenumbers.git@0d6e3e50e17c94262ad1ca3b7d52b11223084bca',
    'src/third_party/libphonenumber/src/resources':
        Var('git_url') + '/external/libphonenumber/resources.git@b6dfdc7952571ff7ee72643cd88c988cbe966396',
    'src/third_party/libphonenumber/src/test':
        Var('git_url') + '/external/libphonenumber/cpp/test.git@f351a7e007f9c9995494499120bbc361ca808a16',
    'src/third_party/libsrtp':
        Var('git_url') + '/chromium/deps/libsrtp.git@6446144c7f083552f21cc4e6768e891bcb767574',
    'src/third_party/libvpx':
        Var('git_url') + '/chromium/deps/libvpx.git@33bbffe8b3fa6d240ab7720f4f46854bd98d7198',
    'src/third_party/libyuv':
        Var('git_url') + '/external/libyuv.git@d204db647e591ccf0e2589236ecea90330d65a66',
    'src/third_party/mesa/src':
        Var('git_url') + '/chromium/deps/mesa.git@071d25db04c23821a12a8b260ab9d96a097402f0',
    'src/third_party/openmax_dl':
        Var('git_url') + '/external/webrtc/deps/third_party/openmax.git@21c8abe416eb1cdf6f759cdce72e715e7f262282',
    'src/third_party/opus/src':
        Var('git_url') + '/chromium/deps/opus.git@cae696156f1e60006e39821e79a1811ae1933c69',
    'src/third_party/pdfium':
        'https://pdfium.googlesource.com/pdfium.git@d10348e27d91ba5525523a3a3491053c8e44f94e',
    'src/third_party/py_trace_event/src':
        Var('git_url') + '/external/py_trace_event.git@dd463ea9e2c430de2b9e53dea57a77b4c3ac9b30',
    'src/third_party/pyftpdlib/src':
        Var('git_url') + '/external/pyftpdlib.git@2be6d65e31c7ee6320d059f581f05ae8d89d7e45',
    'src/third_party/pywebsocket/src':
        Var('git_url') + '/external/pywebsocket/src.git@cb349e87ddb30ff8d1fa1a89be39cec901f4a29c',
    'src/third_party/safe_browsing/testing':
        Var('git_url') + '/external/google-safe-browsing/testing.git@9d7e8064f3ca2e45891470c9b5b1dce54af6a9d6',
    'src/third_party/scons-2.0.1':
        Var('git_url') + '/native_client/src/third_party/scons-2.0.1.git@1c1550e17fc26355d08627fbdec13d8291227067',
    'src/third_party/sfntly/cpp/src':
        Var('git_url') + '/external/sfntly/cpp/src.git@1bdaae8fc788a5ac8936d68bf24f37d977a13dac',
    'src/third_party/skia':
        Var('git_url') + '/skia.git@c7c2314f532bd9cb0c7432330a3670191858ca15',
    'src/third_party/smhasher/src':
        Var('git_url') + '/external/smhasher.git@e87738e57558e0ec472b2fc3a643b838e5b6e88f',
    'src/third_party/snappy/src':
        Var('git_url') + '/external/snappy.git@762bb32f0c9d2f31ba4958c7c0933d22e80c20bf',
    'src/third_party/swig/Lib':
        Var('git_url') + '/chromium/deps/swig/Lib.git@f2a695d52e61e6a8d967731434f165ed400f0d69',
    'src/third_party/trace-viewer':
        Var('git_url') + '/external/trace-viewer.git@a9802a1384185f9c7ee250ce67d3d24d07b11141',
    'src/third_party/usrsctp/usrsctplib':
        Var('git_url') + '/external/usrsctplib.git@13718c7b9fd376fde092cbd3c5347d15059ac652',
    'src/third_party/webdriver/pylib':
        Var('git_url') + '/external/selenium/py.git@5fd78261a75fe08d27ca4835fb6c5ce4b42275bd',
    'src/third_party/webgl/src':
        Var('git_url') + '/external/khronosgroup/webgl.git@0c2bcf36a740181f50ce94a0eaad357219441dee',
    'src/third_party/webpagereplay':
        Var('git_url') + '/external/web-page-replay.git@532b413ff95e8595d5028e0dae75dcf3ba712d2e',
    'src/third_party/webrtc':
        Var('git_url') + '/external/webrtc/trunk/webrtc.git@8b9a856b98700ddb512da12a6e2b7b7dcb2d39ee',
    'src/third_party/yasm/source/patched-yasm':
        Var('git_url') + '/chromium/deps/yasm/patched-yasm.git@4671120cd8558ce62ee8672ebf3eb6f5216f909b',
    'src/tools/deps2git':
        Var('git_url') + '/chromium/tools/deps2git.git@f04828eb0b5acd3e7ad983c024870f17f17b06d9',
    'src/tools/grit':
        Var('git_url') + '/external/grit-i18n.git@a5890a8118c0c80cc0560e6d8d5cf65e5d725509',
    'src/tools/gyp':
        Var('git_url') + '/external/gyp.git@34640080d08ab2a37665512e52142947def3056d',
    'src/tools/page_cycler/acid3':
        Var('git_url') + '/chromium/deps/acid3.git@6be0a66a1ebd7ebc5abc1b2f405a945f6d871521',
    'src/tools/swarming_client':
        Var('git_url') + '/external/swarming.client.git@1b7bfeca33abce319356fd1835a5cd2f74f1916a',
    'src/v8':
        Var('git_url') + '/v8/v8.git@a4a2083f2dbb2c999d04f42393e8669b896e525a',
}

deps_os = {
    'android':
    {
        'src/pdf':
            None,
        'src/third_party/android_protobuf/src':
            Var('git_url') + '/external/android_protobuf.git@94f522f907e3f34f70d9e7816b947e62fddbb267',
        'src/third_party/android_tools':
            Var('git_url') + '/android_tools.git@fd5a8ec0c75d487635f7e6bd3bdc90eb23eba941',
        'src/third_party/apache-mime4j':
            Var('git_url') + '/chromium/deps/apache-mime4j.git@28cb1108bff4b6cf0a2e86ff58b3d025934ebe3a',
        'src/third_party/appurify-python/src':
            Var('git_url') + '/external/github.com/appurify/appurify-python.git@ee7abd5c5ae3106f72b2a0b9d2cb55094688e867',
        'src/third_party/elfutils/src':
            Var('git_url') + '/external/elfutils.git@249673729a7e5dbd5de4f3760bdcaa3d23d154d7',
        'src/third_party/findbugs':
            Var('git_url') + '/chromium/deps/findbugs.git@7f69fa78a6db6dc31866d09572a0e356e921bf12',
        'src/third_party/freetype':
            Var('git_url') + '/chromium/src/third_party/freetype.git@fd6919ac23f74b876c209aba5eaa2be662086391',
        'src/third_party/httpcomponents-client':
            Var('git_url') + '/chromium/deps/httpcomponents-client.git@285c4dafc5de0e853fa845dce5773e223219601c',
        'src/third_party/httpcomponents-core':
            Var('git_url') + '/chromium/deps/httpcomponents-core.git@9f7180a96f8fa5cab23f793c14b413356d419e62',
        'src/third_party/jarjar':
            Var('git_url') + '/chromium/deps/jarjar.git@2e1ead4c68c450e0b77fe49e3f9137842b8b6920',
        'src/third_party/jsr-305/src':
            Var('git_url') + '/external/jsr-305.git@642c508235471f7220af6d5df2d3210e3bfc0919',
        'src/third_party/junit/src':
            Var('git_url') + '/external/junit.git@45a44647e7306262162e1346b750c3209019f2e1',
        'src/third_party/lss':
            Var('git_url') + '/external/linux-syscall-support/lss.git@952107fa7cea0daaabead28c0e92d579bee517eb',
        'src/third_party/mockito/src':
            Var('git_url') + '/external/mockito/mockito.git@ed99a52e94a84bd7c467f2443b475a22fcc6ba8e',
        'src/third_party/requests/src':
            Var('git_url') + '/external/github.com/kennethreitz/requests.git@f172b30356d821d180fa4ecfa3e71c7274a32de4',
        'src/third_party/robolectric/lib':
            Var('git_url') + '/chromium/third_party/robolectric.git@6b63c99a8b6967acdb42cbed0adb067c80efc810',
    },
    'ios':
    {
        'src/chrome/test/data/perf/canvas_bench':
            None,
        'src/chrome/test/data/perf/frame_rate/content':
            None,
        'src/native_client':
            None,
        'src/testing/iossim/third_party/class-dump':
            Var('git_url') + '/chromium/deps/class-dump.git@89bd40883c767584240b4dade8b74e6f57b9bdab',
        'src/third_party/ffmpeg':
            None,
        'src/third_party/gcdwebserver/src':
            Var('git_url') + '/external/github.com/swisspol/GCDWebServer.git@18889793b75d7ee593d62ac88997caad850acdb6',
        'src/third_party/google_toolbox_for_mac/src':
            Var('git_url') + '/external/google-toolbox-for-mac.git@17eee6933bb4a978bf045ef1b12fc68f15b08cd2',
        'src/third_party/hunspell':
            None,
        'src/third_party/hunspell_dictionaries':
            None,
        'src/third_party/nss':
            Var('git_url') + '/chromium/deps/nss.git@bb4e75a43d007518ae7d618665ea2f25b0c60b63',
        'src/third_party/webgl':
            None,
    },
    'mac':
    {
        'src/chrome/installer/mac/third_party/xz/xz':
            Var('git_url') + '/chromium/deps/xz.git@eecaf55632ca72e90eb2641376bce7cdbc7284f7',
        'src/chrome/tools/test/reference_build/chrome_mac':
            Var('git_url') + '/chromium/reference_builds/chrome_mac.git@8dc181329e7c5255f83b4b85dc2f71498a237955',
        'src/third_party/google_toolbox_for_mac/src':
            Var('git_url') + '/external/google-toolbox-for-mac.git@17eee6933bb4a978bf045ef1b12fc68f15b08cd2',
        'src/third_party/lighttpd':
            Var('git_url') + '/chromium/deps/lighttpd.git@9dfa55d15937a688a92cbf2b7a8621b0927d06eb',
        'src/third_party/nss':
            Var('git_url') + '/chromium/deps/nss.git@bb4e75a43d007518ae7d618665ea2f25b0c60b63',
        'src/third_party/pdfsqueeze':
            Var('git_url') + '/external/pdfsqueeze.git@5936b871e6a087b7e50d4cbcb122378d8a07499f',
        'src/third_party/swig/mac':
            Var('git_url') + '/chromium/deps/swig/mac.git@1b182eef16df2b506f1d710b34df65d55c1ac44e',
    },
    'unix':
    {
        'build/third_party/cbuildbot_chromite':
            Var('git_url') + '/chromiumos/chromite.git@881f183fad9526c98e9bf948d7f3179029723462',
        'build/third_party/xvfb':
            Var('git_url') + '/chromium/tools/third_party/xvfb.git@aebb1aadf1422e4d81e831e13746b8f7ae322e07',
        'src/chrome/tools/test/reference_build/chrome_linux':
            Var('git_url') + '/chromium/reference_builds/chrome_linux64.git@033d053a528e820e1de3e2db766678d862a86b36',
        'src/third_party/chromite':
            Var('git_url') + '/chromiumos/chromite.git@fdc9440cb96f8de35202abc285ffb896e04292d3',
        'src/third_party/cros_system_api':
            Var('git_url') + '/chromiumos/platform/system_api.git@e22c1effdfaace2f904536f8a9953644ba90398c',
        'src/third_party/fontconfig/src':
            Var('git_url') + '/external/fontconfig.git@f16c3118e25546c1b749f9823c51827a60aeb5c1',
        'src/third_party/freetype2/src':
            Var('git_url') + '/chromium/src/third_party/freetype2.git@495a23fce9cd125f715dc20643d14fed226d76ac',
        'src/third_party/liblouis/src':
            Var('git_url') + '/external/liblouis-github.git@5f9c03f2a3478561deb6ae4798175094be8a26c2',
        'src/third_party/lss':
            Var('git_url') + '/external/linux-syscall-support/lss.git@952107fa7cea0daaabead28c0e92d579bee517eb',
        'src/third_party/pyelftools':
            Var('git_url') + '/chromiumos/third_party/pyelftools.git@19b3e610c86fcadb837d252c794cb5e8008826ae',
        'src/third_party/stp/src':
            Var('git_url') + '/external/github.com/stp/stp.git@fc94a599207752ab4d64048204f0c88494811b62',
        'src/third_party/swig/linux':
            Var('git_url') + '/chromium/deps/swig/linux.git@866b8e0e0e0cfe99ebe608260030916ca0c3f92d',
        'src/third_party/undoview':
            Var('git_url') + '/chromium/deps/undoview.git@3ba503e248f3cdbd81b78325a24ece0984637559',
        'src/third_party/xdg-utils':
            Var('git_url') + '/chromium/deps/xdg-utils.git@d80274d5869b17b8c9067a1022e4416ee7ed5e0d',
    },
    'win':
    {
        'src/chrome/tools/test/reference_build/chrome_win':
            Var('git_url') + '/chromium/reference_builds/chrome_win.git@f8a3a845dfc845df6b14280f04f86a61959357ef',
        'src/third_party/bison':
            Var('git_url') + '/chromium/deps/bison.git@083c9a45e4affdd5464ee2b224c2df649c6e26c3',
        'src/third_party/cygwin':
            Var('git_url') + '/chromium/deps/cygwin.git@c89e446b273697fadf3a10ff1007a97c0b7de6df',
        'src/third_party/gnu_binutils':
            Var('git_url') + '/native_client/deps/third_party/gnu_binutils.git@f4003433b61b25666565690caf3d7a7a1a4ec436',
        'src/third_party/gperf':
            Var('git_url') + '/chromium/deps/gperf.git@d892d79f64f9449770443fb06da49b5a1e5d33c1',
        'src/third_party/lighttpd':
            Var('git_url') + '/chromium/deps/lighttpd.git@9dfa55d15937a688a92cbf2b7a8621b0927d06eb',
        'src/third_party/mingw-w64/mingw/bin':
            Var('git_url') + '/native_client/deps/third_party/mingw-w64/mingw/bin.git@3cc8b140b883a9fe4986d12cfd46c16a093d3527',
        'src/third_party/nacl_sdk_binaries':
            Var('git_url') + '/chromium/deps/nacl_sdk_binaries.git@759dfca03bdc774da7ecbf974f6e2b84f43699a5',
        'src/third_party/nss':
            Var('git_url') + '/chromium/deps/nss.git@bb4e75a43d007518ae7d618665ea2f25b0c60b63',
        'src/third_party/omaha/src/omaha':
            Var('git_url') + '/external/omaha.git@098c7a3d157218dab4eed595e8f2fbe5a20a0bae',
        'src/third_party/pefile':
            Var('git_url') + '/external/pefile.git@72c6ae42396cb913bcab63c15585dc3b5c3f92f1',
        'src/third_party/perl':
            Var('git_url') + '/chromium/deps/perl.git@ac0d98b5cee6c024b0cffeb4f8f45b6fc5ccdb78',
        'src/third_party/psyco_win32':
            Var('git_url') + '/chromium/deps/psyco_win32.git@f5af9f6910ee5a8075bbaeed0591469f1661d868',
        'src/third_party/swig/win':
            Var('git_url') + '/chromium/deps/swig/win.git@986f013ba518541adf5c839811efb35630a31031',
        'src/third_party/yasm/binaries':
            Var('git_url') + '/chromium/deps/yasm/binaries.git@52f9b3f4b0aa06da24ef8b123058bb61ee468881',
    },
}

include_rules = [
    '+base',
    '+build',
    '+ipc',
    '+library_loaders',
    '+testing',
    '+third_party/icu/source/common/unicode',
    '+third_party/icu/source/i18n/unicode',
    '+url'
]

skip_child_includes = [
    'breakpad',
    'delegate_execute',
    'metro_driver',
    'native_client_sdk',
    'o3d',
    'sdch',
    'skia',
    'testing',
    'third_party',
    'v8',
    'win8'
]

hooks = [
    {
    'action':
         [
    'python',
    'src/build/util/lastchange.py',
    '-o',
    'src/build/util/LASTCHANGE'
],
    'pattern':
         '.',
    'name':
         'lastchange'
},
    {
    'action':
         [
    'python',
    'src/build/util/lastchange.py',
    '-s',
    'src/third_party/WebKit',
    '-o',
    'src/build/util/LASTCHANGE.blink'
],
    'pattern':
         '.',
    'name':
         'lastchange'
},
    {
    'action':
         [
    'python',
    'src/build/gyp_blpwtk2.py'
],
    'pattern':
         '.',
    'name':
         'gyp'
}
]
