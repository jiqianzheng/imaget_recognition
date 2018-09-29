from django import forms

from .utils import get_all_labels




class SearchForm(forms.Form):
    keyword = forms.CharField(widget=forms.TextInput(attrs={"class": "sb-search-input input__field--madoka",
                                                            "type": "search",
                                                            "placeholder": "按类别搜索..."}))

    def clean(self):
        cleaned_data = self.cleaned_data["keyword"]
        kw = cleaned_data
        labels = [item["label"] for item in get_all_labels()]
        if kw not in labels:
            raise forms.ValidationError("类别出错,不在系统数据集类中")
        return kw
