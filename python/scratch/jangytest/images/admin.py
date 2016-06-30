
from django.contrib import admin
from models import Thingy


class column(object):
    def __init__(self, desc="", allow_tags=True):
        self._desc = desc
        self._allow_tags = allow_tags
    def __call__(self, f):
        f.short_description = self._desc
        f.allow_tags = self._allow_tags
        return f

class Nimda(admin.ModelAdmin):
    save_as = False
    save_on_top = False
    actions_on_top = False
    actions_on_bottom = True

class ThingyAdmin(Nimda):
    list_display = ('pk', 'with_mimetype', 'with_image', 'w', 'h')
    readonly_fields = ('pk', 'createdate', 'modifydate', 'w', 'h')
    fieldsets = [
        (None, {
            'fields': [
                'image',
                ('w', 'h')
            ]
        }),
        ("Etc.", {
            'fields': [
                'createdate',
                'modifydate',
            ],
            'classes': ['collapse'],
        }),
    ]
    
    @column(desc="Uploaded Image File")
    def with_image(self, obj):
        return u"""
            <nobr>
                <b><a href="%(edit)s">
                    <img src="%(title)s" style="display: none; border: 2px solid black;">
                    %(title)s
                    </a></b>
            </nobr>
        """ % dict(
            edit="./%s/" % obj.pk,
            title=obj.image)
    
    @column(desc="Image MIME Type")
    def with_mimetype(self, obj):
        return u"""
            <nobr>
                <a href="%(edit)s">
                    %(title)s
                    </a>
            </nobr>
        """ % dict(
            edit="./%s/" % obj.pk,
            title=obj.mimetype)
    


admin.site.register(Thingy, ThingyAdmin)

