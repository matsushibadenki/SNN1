# matsushibadenki/snn/app/utils.py
# Gradioアプリケーション用の共通ユーティリティ
#
# 機能:
# - アプリケーション間で共有される定数や関数を定義する。
# - UI用のアバターSVGアイコンを一元管理する。

def get_avatar_svgs():
    """
    Gradioチャットボット用のアバターSVGアイコンのタプルを返す。

    Returns:
        tuple[str, str]: ユーザー用とアシスタント用のSVGアイコン文字列のタプル。
    """
    user_avatar_svg = r"""
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-user"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
    """
    assistant_avatar_svg = r"""
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-zap"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
    """
    return user_avatar_svg, assistant_avatar_svg
