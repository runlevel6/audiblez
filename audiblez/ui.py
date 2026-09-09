#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# A simple wxWidgets UI for audiblez

import torch.cuda
import numpy as np
import soundfile
import threading
import platform
import subprocess
import io
import os
import wx
from wx.lib.newevent import NewEvent
from wx.lib.scrolledpanel import ScrolledPanel
from PIL import Image
from tempfile import NamedTemporaryFile
from pathlib import Path

# fix #9: import settings helpers from core — single source of truth
from audiblez.core import load_settings, save_settings, DEFAULT_VOICE, check_phonetic_transcription_ai, correct_phonetics_ai
from audiblez.voices import voices, flags

EVENTS = {
    'CORE_STARTED': NewEvent(),
    'CORE_PROGRESS': NewEvent(),
    'CORE_CHAPTER_STARTED': NewEvent(),
    'CORE_CHAPTER_FINISHED': NewEvent(),
    'CORE_AI_REWRITE': NewEvent(),
    'CORE_AI_RETRY_EXHAUSTED': NewEvent(),
    'CORE_FINISHED': NewEvent()
}

border = 5


class MainWindow(wx.Frame):
    def __init__(self, parent, title):
        screen_width, screen_h = wx.GetDisplaySize()
        self.window_width = int(screen_width * 0.6)
        super().__init__(parent, title=title, size=(self.window_width, self.window_width * 3 // 4))
        self.chapters_panel = None
        self.preview_threads = []
        self.selected_chapter = None
        self.selected_book = None
        self.synthesis_in_progress = False
        self.stop_event = None

        self.Bind(EVENTS['CORE_STARTED'][1], self.on_core_started)
        self.Bind(EVENTS['CORE_CHAPTER_STARTED'][1], self.on_core_chapter_started)
        self.Bind(EVENTS['CORE_CHAPTER_FINISHED'][1], self.on_core_chapter_finished)
        self.Bind(EVENTS['CORE_PROGRESS'][1], self.on_core_progress)
        self.Bind(EVENTS['CORE_AI_REWRITE'][1], self.on_core_ai_rewrite)
        self.Bind(EVENTS['CORE_AI_RETRY_EXHAUSTED'][1], self.on_core_ai_retry_exhausted)
        self.Bind(EVENTS['CORE_FINISHED'][1], self.on_core_finished)

        self.settings = load_settings()
        self.last_open_dir = self.settings.get('last_open_dir', str(Path.home()))

        self.create_menu()
        self.create_layout()
        self.Centre()
        self.Show(True)

    def create_menu(self):
        menubar = wx.MenuBar()
        file_menu = wx.Menu()

        open_item = wx.MenuItem(file_menu, wx.ID_OPEN, "&Open\tCtrl+O")
        file_menu.Append(open_item)
        self.Bind(wx.EVT_MENU, self.on_open, open_item)

        exit_item = wx.MenuItem(file_menu, wx.ID_EXIT, "&Exit\tCtrl+Q")
        file_menu.Append(exit_item)
        self.Bind(wx.EVT_MENU, self.on_exit, exit_item)

        menubar.Append(file_menu, "&File")
        self.SetMenuBar(menubar)

    def on_core_started(self, event):
        print('CORE_STARTED')
        self.progress_bar_label.Show()
        self.progress_bar.Show()
        self.progress_bar.SetValue(0)
        self.progress_bar.Layout()
        self.eta_label.Show()
        self.params_panel.Layout()
        self.synth_panel.Layout()

    def on_core_chapter_started(self, event):
        self.set_table_chapter_status(event.chapter_index, "⏳ In Progress")

    def on_core_chapter_finished(self, event):
        self.set_table_chapter_status(event.chapter_index, "✅ Done")   

    def on_core_progress(self, event):
        self.progress_bar.SetValue(event.stats.progress)
        self.progress_bar_label.SetLabel(f"Synthesis Progress: {event.stats.progress}%")
        self.eta_label.SetLabel(f"Estimated Time Remaining: {event.stats.eta}")
        self.synth_panel.Layout()

    def on_core_ai_rewrite(self, event):
        ci = getattr(event, 'chapter_index', None)
        ct = getattr(event, 'chapter_total', None)
        idx = getattr(event, 'chunk_index', 0)
        tot = getattr(event, 'chunk_total', 0)
        if ci is not None and tot > 0:
            self.progress_bar_label.SetLabel(
                f"AI rewriting chapter {ci + 1}/{ct}: chunk {idx}/{tot}")
        elif ci is not None:
            self.progress_bar_label.SetLabel(f"AI rewriting chapter {ci + 1}/{ct}…")
        else:
            self.progress_bar_label.SetLabel("AI rewriting…")
        self.synth_panel.Layout()

    def on_core_ai_retry_exhausted(self, event):
        msg = getattr(event, 'message', 'AI service unavailable after multiple retries.')
        answer = wx.MessageBox(
            f"The Gemini AI service is currently unavailable or experiencing high demand.\n\n"
            f"All retry attempts have been exhausted.\n\n"
            f"Details: {msg}\n\n"
            f"Click OK to continue synthesis without AI phonetic correction, "
            f"or Cancel to stop the synthesis.",
            "AI Service Unavailable",
            wx.OK | wx.CANCEL | wx.ICON_WARNING
        )
        if answer == wx.CANCEL and self.synthesis_in_progress:
            self.cancel_current_synthesis()

    def on_core_finished(self, event):
        self.synthesis_in_progress = False
        self.cancel_button.Hide()
        self.start_button.Enable()
        self.synth_panel.Layout()
        self.save_current_settings()
        self.open_folder_with_explorer(self.output_folder_text_ctrl.GetValue())

    def create_layout(self):
        top_panel = wx.Panel(self)
        top_sizer = wx.BoxSizer(wx.HORIZONTAL)
        top_panel.SetSizer(top_sizer)

        open_epub_button = wx.Button(top_panel, label="📁 Open EPUB")
        open_epub_button.Bind(wx.EVT_BUTTON, self.on_open)
        top_sizer.Add(open_epub_button, 0, wx.ALL, 5)

        help_button = wx.Button(top_panel, label="ℹ️ About")
        help_button.Bind(wx.EVT_BUTTON, lambda event: self.about_dialog())
        top_sizer.Add(help_button, 0, wx.ALL, 5)

        self.main_sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.main_sizer)

        self.splitter = wx.Panel(self)
        self.splitter_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.splitter.SetSizer(self.splitter_sizer)

        self.main_sizer.Add(top_panel, 0, wx.ALL | wx.EXPAND, 5)
        self.main_sizer.Add(self.splitter, 1, wx.EXPAND)

    def create_layout_for_ebook(self, splitter):
        splitter_left = wx.Panel(splitter, -1)
        splitter_right = wx.Panel(self.splitter)
        self.splitter_left, self.splitter_right = splitter_left, splitter_right
        self.splitter_sizer.Add(splitter_left, 1, wx.ALL | wx.EXPAND, 5)
        self.splitter_sizer.Add(splitter_right, 2, wx.ALL | wx.EXPAND, 5)

        self.left_sizer = wx.BoxSizer(wx.VERTICAL)
        splitter_left.SetSizer(self.left_sizer)

        self.center_panel = wx.Panel(splitter_right)
        self.center_sizer = wx.BoxSizer(wx.VERTICAL)
        self.center_panel.SetSizer(self.center_sizer)
        self.text_area = wx.TextCtrl(self.center_panel, style=wx.TE_MULTILINE, size=(int(self.window_width * 0.4), -1))
        font = wx.Font(14, wx.MODERN, wx.NORMAL, wx.NORMAL)
        self.text_area.SetFont(font)
        self.text_area.Bind(wx.EVT_TEXT, lambda event: setattr(self.selected_chapter, 'extracted_text', self.text_area.GetValue()))

        self.chapter_label = wx.StaticText(
            self.center_panel, label=f'Edit / Preview content for section "{self.selected_chapter.short_name}":')
        preview_button = wx.Button(self.center_panel, label="🔊 Preview")
        preview_button.Bind(wx.EVT_BUTTON, self.on_preview_chapter)

        check_ai_button = wx.Button(self.center_panel, label="🤖 Check with AI")
        check_ai_button.Bind(wx.EVT_BUTTON, self.on_check_phonetic_ai)

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(preview_button, 0, wx.ALL, 5)
        button_sizer.Add(check_ai_button, 0, wx.ALL, 5)

        self.center_sizer.Add(self.chapter_label, 0, wx.ALL, 5)
        self.center_sizer.Add(button_sizer, 0, wx.ALL, 5)
        self.center_sizer.Add(self.text_area, 1, wx.ALL | wx.EXPAND, 5)

        splitter_right_sizer = wx.BoxSizer(wx.HORIZONTAL)
        splitter_right.SetSizer(splitter_right_sizer)

        self.create_right_panel(splitter_right)
        splitter_right_sizer.Add(self.center_panel, 1, wx.ALL | wx.EXPAND, 5)
        splitter_right_sizer.Add(self.right_panel, 1, wx.ALL | wx.EXPAND, 5)

    def about_dialog(self):
        msg = ("A simple tool to generate audiobooks from EPUB files using Kokoro-82M models\n"
               "Distributed under the MIT License.\n\n"
               "Originally by Claudio Santini 2025 — https://claudio.uk\n"
               "Fork by Vlad Reshetov 2025\n")
        wx.MessageBox(msg, "Audiblez")

    def create_right_panel(self, splitter_right):
        self.right_panel = wx.Panel(splitter_right)
        self.right_sizer = wx.BoxSizer(wx.VERTICAL)
        self.right_panel.SetSizer(self.right_sizer)

        self.book_info_panel_box = wx.Panel(self.right_panel, style=wx.SUNKEN_BORDER)
        book_info_panel_box_sizer = wx.StaticBoxSizer(wx.VERTICAL, self.book_info_panel_box, "Book Details")
        self.book_info_panel_box.SetSizer(book_info_panel_box_sizer)
        self.right_sizer.Add(self.book_info_panel_box, 1, wx.ALL | wx.EXPAND, 5)

        self.book_info_panel = wx.Panel(self.book_info_panel_box, style=wx.BORDER_NONE)
        self.book_info_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.book_info_panel.SetSizer(self.book_info_sizer)
        book_info_panel_box_sizer.Add(self.book_info_panel, 1, wx.ALL | wx.EXPAND, 5)

        self.cover_bitmap = wx.StaticBitmap(self.book_info_panel, -1)
        self.book_info_sizer.Add(self.cover_bitmap, 0, wx.ALL, 5)
        self.cover_bitmap.Refresh()
        self.book_info_panel.Refresh()
        self.book_info_panel.Layout()
        self.cover_bitmap.Layout()

        self.create_book_details_panel()
        self.create_params_panel()
        self.create_synthesis_panel()

    def create_book_details_panel(self):
        book_details_panel = wx.Panel(self.book_info_panel)
        book_details_sizer = wx.GridBagSizer(10, 10)
        book_details_panel.SetSizer(book_details_sizer)
        self.book_info_sizer.Add(book_details_panel, 1, wx.ALL | wx.EXPAND, 5)

        title_label = wx.StaticText(book_details_panel, label="Title:")
        title_text = wx.StaticText(book_details_panel, label=getattr(self, 'selected_book_title', 'N/A'))
        book_details_sizer.Add(title_label, pos=(0, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(title_text, pos=(0, 1), flag=wx.ALL, border=5)

        author_label = wx.StaticText(book_details_panel, label="Author:")
        author_text = wx.StaticText(book_details_panel, label=getattr(self, 'selected_book_author', 'N/A'))
        book_details_sizer.Add(author_label, pos=(1, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(author_text, pos=(1, 1), flag=wx.ALL, border=5)

        length_label = wx.StaticText(book_details_panel, label="Total Length:")
        total_len = sum(len(c.extracted_text) for c in self.document_chapters) if hasattr(self, 'document_chapters') else 0
        length_text = wx.StaticText(book_details_panel, label=f'{total_len:,} characters')
        book_details_sizer.Add(length_label, pos=(2, 0), flag=wx.ALL, border=5)
        book_details_sizer.Add(length_text, pos=(2, 1), flag=wx.ALL, border=5)

    def create_params_panel(self):
        panel_box = wx.Panel(self.right_panel, style=wx.SUNKEN_BORDER)
        panel_box_sizer = wx.StaticBoxSizer(wx.VERTICAL, panel_box, "Audiobook Parameters")
        panel_box.SetSizer(panel_box_sizer)

        panel = self.params_panel = wx.Panel(panel_box)
        panel_box_sizer.Add(panel, 1, wx.ALL | wx.EXPAND, 5)
        self.right_sizer.Add(panel_box, 1, wx.ALL | wx.EXPAND, 5)
        sizer = wx.GridBagSizer(10, 10)
        panel.SetSizer(sizer)

        engine_label = wx.StaticText(panel, label="Engine:")
        engine_radio_panel = wx.Panel(panel)
        cpu_radio = wx.RadioButton(engine_radio_panel, label="CPU", style=wx.RB_GROUP)
        cuda_radio = wx.RadioButton(engine_radio_panel, label="CUDA")
        if torch.cuda.is_available():
            cuda_radio.SetValue(True)
        else:
            cpu_radio.SetValue(True)
        sizer.Add(engine_label, pos=(0, 0), flag=wx.ALL, border=border)
        sizer.Add(engine_radio_panel, pos=(0, 1), flag=wx.ALL, border=border)
        engine_radio_panel_sizer = wx.BoxSizer(wx.HORIZONTAL)
        engine_radio_panel.SetSizer(engine_radio_panel_sizer)
        engine_radio_panel_sizer.Add(cpu_radio, 0, wx.ALL, 5)
        engine_radio_panel_sizer.Add(cuda_radio, 0, wx.ALL, 5)
        cpu_radio.Bind(wx.EVT_RADIOBUTTON, lambda event: torch.set_default_device('cpu'))
        cuda_radio.Bind(wx.EVT_RADIOBUTTON, lambda event: torch.set_default_device('cuda'))

        flag_and_voice_list = []
        for code, l in voices.items():
            for v in l:
                flag_and_voice_list.append(f'{flags[code]} {v}')

        voice_label = wx.StaticText(panel, label="Voice:")
        default_voice_from_settings = self.settings.get('voice', DEFAULT_VOICE)
        initial_voice_display = next(
            (fv for fv in flag_and_voice_list if fv.endswith(default_voice_from_settings)),
            flag_and_voice_list[0] if flag_and_voice_list else ''
        )
        self.selected_voice = initial_voice_display
        voice_dropdown = wx.ComboBox(panel, choices=flag_and_voice_list, value=initial_voice_display)
        voice_dropdown.Bind(wx.EVT_COMBOBOX, self.on_select_voice)
        sizer.Add(voice_label, pos=(1, 0), flag=wx.ALL, border=border)
        sizer.Add(voice_dropdown, pos=(1, 1), flag=wx.ALL, border=border)

        speed_label = wx.StaticText(panel, label="Speed:")
        speed_text_input = wx.TextCtrl(panel, value=str(self.settings.get('speed', 1.0)))
        self.selected_speed = float(speed_text_input.GetValue())
        speed_text_input.Bind(wx.EVT_TEXT, self.on_select_speed)
        sizer.Add(speed_label, pos=(2, 0), flag=wx.ALL, border=border)
        sizer.Add(speed_text_input, pos=(2, 1), flag=wx.ALL, border=border)

        ai_enabled_label = wx.StaticText(panel, label="AI Phonetic Check:")
        ai_enabled_checkbox = wx.CheckBox(panel, label="Enabled")
        ai_enabled_checkbox.SetValue(self.settings.get('gemini_enabled', False))
        ai_enabled_checkbox.Bind(wx.EVT_CHECKBOX, self.on_ai_enabled_changed)
        self.ai_enabled_checkbox = ai_enabled_checkbox
        sizer.Add(ai_enabled_label, pos=(3, 0), flag=wx.ALL, border=border)
        sizer.Add(ai_enabled_checkbox, pos=(3, 1), flag=wx.ALL, border=border)

        api_key_label = wx.StaticText(panel, label="Gemini API Key:")
        api_key_text_input = wx.TextCtrl(panel, value=self.settings.get('gemini_api_key', ''), style=wx.TE_PASSWORD)
        api_key_text_input.Bind(wx.EVT_TEXT, self.on_ai_api_key_changed)
        self.ai_api_key_text_ctrl = api_key_text_input
        sizer.Add(api_key_label, pos=(4, 0), flag=wx.ALL, border=border)
        sizer.Add(api_key_text_input, pos=(4, 1), flag=wx.ALL | wx.EXPAND, border=border)

        ai_model_label = wx.StaticText(panel, label="Gemini Model:")
        ai_model_choices = ['gemini-3.1-flash-lite', 'gemini-3.5-flash', 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-flash-lite-latest']
        ai_model_dropdown = wx.ComboBox(panel, choices=ai_model_choices, value=self.settings.get('gemini_model', 'gemini-3.1-flash-lite'))
        ai_model_dropdown.Bind(wx.EVT_COMBOBOX, self.on_ai_model_changed)
        self.ai_model_dropdown = ai_model_dropdown
        sizer.Add(ai_model_label, pos=(5, 0), flag=wx.ALL, border=border)
        sizer.Add(ai_model_dropdown, pos=(5, 1), flag=wx.ALL | wx.EXPAND, border=border)

        output_folder_label = wx.StaticText(panel, label="Output Folder:")
        initial_output_folder = self.settings.get('output_folder', os.path.abspath('.'))
        self.output_folder_text_ctrl = wx.TextCtrl(panel, value=initial_output_folder)
        self.output_folder_text_ctrl.SetEditable(False)
        output_folder_button = wx.Button(panel, label="📂 Select")
        output_folder_button.Bind(wx.EVT_BUTTON, self.open_output_folder_dialog)
        sizer.Add(output_folder_label, pos=(6, 0), flag=wx.ALL, border=border)
        sizer.Add(self.output_folder_text_ctrl, pos=(6, 1), flag=wx.ALL | wx.EXPAND, border=border)
        sizer.Add(output_folder_button, pos=(7, 1), flag=wx.ALL, border=border)

    def create_synthesis_panel(self):
        panel_box = wx.Panel(self.right_panel, style=wx.SUNKEN_BORDER)
        panel_box_sizer = wx.StaticBoxSizer(wx.VERTICAL, panel_box, "Audiobook Generation Status")
        panel_box.SetSizer(panel_box_sizer)

        panel = self.synth_panel = wx.Panel(panel_box)
        panel_box_sizer.Add(panel, 1, wx.ALL | wx.EXPAND, 5)
        self.right_sizer.Add(panel_box, 1, wx.ALL | wx.EXPAND, 5)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)

        self.start_button = wx.Button(panel, label="🚀 Start Audiobook Synthesis")
        self.start_button.Bind(wx.EVT_BUTTON, self.on_start)
        sizer.Add(self.start_button, 0, wx.ALL, 5)

        self.cancel_button = wx.Button(panel, label="⛔ Cancel Synthesis")
        self.cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)
        self.cancel_button.Hide()
        sizer.Add(self.cancel_button, 0, wx.ALL, 5)

        self.progress_bar_label = wx.StaticText(panel, label="Synthesis Progress:")
        sizer.Add(self.progress_bar_label, 0, wx.ALL, 5)
        self.progress_bar = wx.Gauge(panel, range=100, style=wx.GA_PROGRESS)
        self.progress_bar.SetMinSize((-1, 30))
        sizer.Add(self.progress_bar, 0, wx.ALL | wx.EXPAND, 5)
        self.progress_bar_label.Hide()
        self.progress_bar.Hide()

        self.eta_label = wx.StaticText(panel, label="Estimated Time Remaining: ")
        self.eta_label.Hide()
        sizer.Add(self.eta_label, 0, wx.ALL, 5)

    def open_output_folder_dialog(self, event):
        with wx.DirDialog(self, "Choose a directory:", style=wx.DD_DEFAULT_STYLE) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            output_folder = dialog.GetPath()
            print(f"Selected output folder: {output_folder}")
            self.output_folder_text_ctrl.SetValue(output_folder)
            self.save_current_settings()

    def on_select_voice(self, event):
        self.selected_voice = event.GetString()
        self.save_current_settings()

    def on_select_speed(self, event):
        try:
            speed = float(event.GetString())
            if speed > 0:
                print('Selected speed', speed)
                self.selected_speed = speed
                self.save_current_settings()
            else:
                print("Speed must be a positive number.")
        except ValueError:
            print("Invalid speed value. Please enter a number.")

    def on_ai_enabled_changed(self, event):
        self.save_current_settings()

    def on_ai_api_key_changed(self, event):
        self.save_current_settings()

    def on_ai_model_changed(self, event):
        self.save_current_settings()

    def save_current_settings(self):
        """Save current GUI settings via core's save_settings."""
        output_folder = self.output_folder_text_ctrl.GetValue() if (hasattr(self, 'output_folder_text_ctrl') and self.output_folder_text_ctrl) else self.settings.get('output_folder', '.')
        voice = self.get_selected_voice() if hasattr(self, 'selected_voice') else self.settings.get('voice', 'af_heart')
        speed = self.selected_speed if hasattr(self, 'selected_speed') else self.settings.get('speed', 1.0)
        gemini_api_key = self.ai_api_key_text_ctrl.GetValue() if (hasattr(self, 'ai_api_key_text_ctrl') and self.ai_api_key_text_ctrl) else self.settings.get('gemini_api_key', '')
        gemini_api_key = ' '.join(gemini_api_key.replace('\t', ' ').replace('\n', ' ').replace('\r', ' ').split())
        gemini_model = self.ai_model_dropdown.GetValue() if (hasattr(self, 'ai_model_dropdown') and self.ai_model_dropdown) else self.settings.get('gemini_model', 'gemini-3.1-flash-lite')
        gemini_enabled = self.ai_enabled_checkbox.GetValue() if (hasattr(self, 'ai_enabled_checkbox') and self.ai_enabled_checkbox) else self.settings.get('gemini_enabled', False)
        last_open_dir = self.last_open_dir if hasattr(self, 'last_open_dir') else self.settings.get('last_open_dir', '')

        save_settings(
            output_folder=output_folder,
            voice=voice,
            speed=speed,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            gemini_enabled=gemini_enabled,
            last_open_dir=last_open_dir,
        )

    def open_epub(self, file_path):
        if hasattr(self, 'selected_book'):
            ai_enabled = self.ai_enabled_checkbox.GetValue() if (hasattr(self, 'ai_enabled_checkbox') and self.ai_enabled_checkbox) else self.settings.get('gemini_enabled', False)
            self.splitter.DestroyChildren()
        else:
            ai_enabled = self.settings.get('gemini_enabled', False)

        self.selected_file_path = file_path
        print(f"Opening file: {file_path}")

        import audiblez.core as core
        from ebooklib import epub

        try:
            book = epub.read_epub(file_path)
        except Exception as e:
            wx.MessageBox(f"Error opening EPUB file: {e}", "Error", wx.OK | wx.ICON_ERROR)
            print(f"Error reading EPUB file '{file_path}': {e}")
            return

        meta_title = book.get_metadata('DC', 'title')
        self.selected_book_title = meta_title[0][0] if meta_title else ''
        meta_creator = book.get_metadata('DC', 'creator')
        self.selected_book_author = meta_creator[0][0] if meta_creator else ''
        self.selected_book = book

        self.document_chapters = core.find_document_chapters_and_extract_texts(book, ai_enabled=ai_enabled)
        good_chapters = core.find_good_chapters(self.document_chapters)
        self.selected_chapter = good_chapters[0] if good_chapters else None
        if self.selected_chapter is None:
            wx.MessageBox("No readable chapters found in this EPUB.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        for chapter in self.document_chapters:
            chapter.short_name = (chapter.get_name()
                                  .replace('.xhtml', '').replace('xhtml/', '')
                                  .replace('.html', '').replace('Text/', ''))
            chapter.is_selected = chapter in good_chapters

        self.create_layout_for_ebook(self.splitter)

        cover = core.find_cover(book)
        if cover is not None:
            pil_image = Image.open(io.BytesIO(cover.content))
            wx_img = wx.EmptyImage(pil_image.size[0], pil_image.size[1])
            wx_img.SetData(pil_image.convert("RGB").tobytes())
            cover_h = 200
            cover_w = int(cover_h * pil_image.size[0] / pil_image.size[1])
            wx_img.Rescale(cover_w, cover_h)
            self.cover_bitmap.SetBitmap(wx_img.ConvertToBitmap())
            self.cover_bitmap.SetMaxSize((200, cover_h))

        chapters_panel = self.create_chapters_table_panel(good_chapters)

        if self.chapters_panel:
            self.left_sizer.Replace(self.chapters_panel, chapters_panel)
            self.chapters_panel.Destroy()
            self.chapters_panel = chapters_panel
        else:
            self.left_sizer.Add(chapters_panel, 1, wx.ALL | wx.EXPAND, 5)
            self.chapters_panel = chapters_panel

        self.splitter_left.Layout()
        self.splitter_right.Layout()
        self.splitter.Layout()

        if self.selected_chapter:
            self.text_area.SetValue(self.selected_chapter.extracted_text)
            self.chapter_label.SetLabel(f'Edit / Preview content for section "{self.selected_chapter.short_name}":')

    def on_table_checked(self, event):
        self.document_chapters[event.GetIndex()].is_selected = True

    def on_table_unchecked(self, event):
        self.document_chapters[event.GetIndex()].is_selected = False

    def on_table_selected(self, event):
        chapter = self.document_chapters[event.GetIndex()]
        print('Selected', event.GetIndex(), chapter.short_name)
        self.selected_chapter = chapter
        self.text_area.SetValue(chapter.extracted_text)
        self.chapter_label.SetLabel(f'Edit / Preview content for section "{chapter.short_name}":')

    def create_chapters_table_panel(self, good_chapters):
        panel = ScrolledPanel(self.splitter_left, -1, style=wx.TAB_TRAVERSAL | wx.SUNKEN_BORDER)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)

        self.table = table = wx.ListCtrl(panel, style=wx.LC_REPORT | wx.BORDER_SUNKEN)
        table.InsertColumn(0, "Included")
        table.InsertColumn(1, "Chapter Name")
        table.InsertColumn(2, "Chapter Length")
        table.InsertColumn(3, "Status")
        table.SetColumnWidth(0, 80)
        table.SetColumnWidth(1, 150)
        table.SetColumnWidth(2, 150)
        table.SetColumnWidth(3, 100)
        table.SetSize((250, -1))
        table.EnableCheckBoxes()
        table.Bind(wx.EVT_LIST_ITEM_CHECKED, self.on_table_checked)
        table.Bind(wx.EVT_LIST_ITEM_UNCHECKED, self.on_table_unchecked)
        table.Bind(wx.EVT_LIST_ITEM_SELECTED, self.on_table_selected)

        for i, chapter in enumerate(self.document_chapters):
            auto_selected = chapter in good_chapters
            table.Append(['', chapter.short_name, f"{len(chapter.extracted_text):,}"])
            if auto_selected:
                table.CheckItem(i)

        title_text = wx.StaticText(panel, label="Select chapters to include in the audiobook:")
        sizer.Add(title_text, 0, wx.ALL, 5)
        sizer.Add(table, 1, wx.ALL | wx.EXPAND, 5)
        return panel

    def get_selected_voice(self):
        """Return just the voice code, stripping any leading flag emoji."""
        parts = self.selected_voice.split(' ')
        return parts[1] if len(parts) > 1 else self.selected_voice

    def get_selected_speed(self):
        return float(self.selected_speed)

    def on_preview_chapter(self, event):
        if not self.selected_chapter:
            wx.MessageBox("No chapter selected for preview.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        voice = self.get_selected_voice()
        button = event.GetEventObject()
        ai_enabled = self.ai_enabled_checkbox.GetValue()
        api_key = self.ai_api_key_text_ctrl.GetValue().strip() if ai_enabled else ''
        model = self.ai_model_dropdown.GetValue()

        if ai_enabled and not api_key:
            wx.MessageBox("AI phonetic check is enabled but the Gemini API key is missing. "
                          "Disable AI or enter a key in the settings panel.",
                          "Missing API Key", wx.OK | wx.ICON_INFORMATION)
            return

        text = self.selected_chapter.extracted_text[:300]
        if not text.strip():
            wx.MessageBox("Selected chapter has no text for preview.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        def reset_button():
            wx.CallAfter(button.SetLabel, "🔊 Preview")
            wx.CallAfter(button.Enable)

        button.SetLabel("⏳")
        button.Disable()

        def generate_preview():
            import audiblez.core as core
            from kokoro import KPipeline
            try:
                preview_text = text
                if ai_enabled:
                    corrected = core.correct_phonetics_ai(
                        text=preview_text, api_key=api_key, model=model)
                    if corrected and corrected.strip() and corrected != preview_text:
                        preview_text = corrected

                pipeline = KPipeline(lang_code=core.lang_code_from_voice(voice))
                audio_segments = core.gen_audio_segments(
                    pipeline, preview_text, voice=voice, speed=self.get_selected_speed())

                if not audio_segments:
                    wx.CallAfter(wx.MessageBox, "Could not generate audio for preview.", "Error", wx.OK | wx.ICON_ERROR)
                    return

                final_audio = np.concatenate(audio_segments)
                with NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                    soundfile.write(tmp, final_audio, core.sample_rate)
                    tmp_path = tmp.name

                subprocess.run(['ffplay', '-autoexit', '-nodisp', tmp_path])
                os.remove(tmp_path)

            except Exception as e:
                wx.CallAfter(wx.MessageBox, f"Error during preview: {e}", "Preview Error", wx.OK | wx.ICON_ERROR)
                import traceback
                traceback.print_exc()
            finally:
                reset_button()

        # fix #8: don't join on the UI thread — use daemon threads and just
        # let previous previews finish in the background.  The finally block
        # above re-enables the button when each thread ends naturally.
        thread = threading.Thread(target=generate_preview, daemon=True)
        thread.start()
        self.preview_threads.append(thread)
        # Prune dead threads from the list so it doesn't grow forever
        self.preview_threads = [t for t in self.preview_threads if t.is_alive()]

    def on_check_phonetic_ai(self, event):
        if not self.selected_chapter:
            wx.MessageBox("No chapter selected for AI check.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        if not self.ai_enabled_checkbox.GetValue():
            wx.MessageBox("AI phonetic check is not enabled. Please enable it in the settings panel.",
                          "AI Not Enabled", wx.OK | wx.ICON_INFORMATION)
            return

        api_key = self.ai_api_key_text_ctrl.GetValue().strip()
        if not api_key:
            wx.MessageBox("Gemini API key is missing. Please enter it in the settings panel.",
                          "Missing API Key", wx.OK | wx.ICON_INFORMATION)
            return

        text = self.selected_chapter.extracted_text.strip()
        if not text:
            wx.MessageBox("Selected chapter has no text for AI analysis.", "Warning", wx.OK | wx.ICON_WARNING)
            return

        model = self.ai_model_dropdown.GetValue()

        button = event.GetEventObject()
        button.SetLabel("⏳")
        button.Disable()

        def run_ai_check():
            import audiblez.core as core
            try:
                result = core.check_phonetic_transcription_ai(
                    text=text,
                    api_key=api_key,
                    model=model
                )
                wx.CallAfter(self.show_ai_result_dialog, result)
            except Exception as e:
                wx.CallAfter(wx.MessageBox, f"Error during AI check: {e}", "AI Error", wx.OK | wx.ICON_ERROR)
                import traceback
                traceback.print_exc()
            finally:
                wx.CallAfter(button.SetLabel, "🤖 Check with AI")
                wx.CallAfter(button.Enable)

        thread = threading.Thread(target=run_ai_check, daemon=True)
        thread.start()

    def show_ai_result_dialog(self, result_text):
        dialog = wx.Dialog(self, title="AI Phonetic Transcription Analysis", size=(700, 500))
        sizer = wx.BoxSizer(wx.VERTICAL)

        text_ctrl = wx.TextCtrl(dialog, style=wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP)
        text_ctrl.SetValue(result_text)

        close_button = wx.Button(dialog, label="Close")
        close_button.Bind(wx.EVT_BUTTON, lambda event: dialog.EndModal(wx.ID_OK))

        sizer.Add(text_ctrl, 1, wx.ALL | wx.EXPAND, 10)
        sizer.Add(close_button, 0, wx.ALL | wx.CENTER, 10)

        dialog.SetSizer(sizer)
        dialog.ShowModal()
        dialog.Destroy()

    def on_start(self, event):
        self.synthesis_in_progress = True
        file_path = self.selected_file_path
        voice = self.get_selected_voice()
        speed = self.get_selected_speed()
        selected_chapters = [c for c in self.document_chapters if c.is_selected]

        if not selected_chapters:
            wx.MessageBox("No chapters selected. Please select at least one chapter.",
                          "No Chapters Selected", wx.OK | wx.ICON_WARNING)
            self.synthesis_in_progress = False
            return

        if not file_path:
            wx.MessageBox("No EPUB file loaded. Please open an EPUB file first.",
                          "No File Loaded", wx.OK | wx.ICON_WARNING)
            self.synthesis_in_progress = False
            return

        self.start_button.Disable()
        self.cancel_button.Show()
        self.synth_panel.Layout()
        self.params_panel.Disable()
        self.table.EnableCheckBoxes(False)

        self.stop_event = threading.Event()
        print('Starting Audiobook Synthesis', dict(file_path=file_path, voice=voice, speed=speed))
        self.core_thread = CoreThread(
            params=dict(file_path=file_path, voice=voice, pick_manually=False, speed=speed,
                        output_folder=self.output_folder_text_ctrl.GetValue(),
                        selected_chapters=selected_chapters),
            stop_event=self.stop_event)
        self.core_thread.start()

    def on_open(self, event):
        with wx.FileDialog(self, "Open EPUB File", wildcard="*.epub",
                           defaultDir=self.last_open_dir,
                           style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST) as dialog:
            if dialog.ShowModal() == wx.ID_CANCEL:
                return
            file_path = dialog.GetPath()
            if not file_path:
                return
            self.last_open_dir = str(Path(file_path).parent)
            self.save_current_settings()
            if self.synthesis_in_progress:
                wx.MessageBox("Audiobook synthesis is still in progress. Please wait.",
                              "Synthesis in Progress")
            else:
                wx.CallAfter(self.open_epub, file_path)

    def on_cancel(self, event):
        self.cancel_current_synthesis()

    def cancel_current_synthesis(self):
        """Stop the current run: fire stop_event and flag the UI as stopped.

        Shared by the "⛔ Cancel Synthesis" button and the Cancel button on the
        AI-service-unavailable popup (on_core_ai_retry_exhausted).
        """
        if self.stop_event:
            self.stop_event.set()
        self.cancel_button.Disable()
        self.cancel_button.SetLabel("⏳ Stopping…")
        self.synthesis_in_progress = False

    def on_exit(self, event):
        if self.synthesis_in_progress:
            answer = wx.MessageBox(
                "Audiobook synthesis is still in progress.\nStop synthesis and exit?",
                "Exit Audiblez", wx.YES_NO | wx.ICON_WARNING)
            if answer != wx.YES:
                return
            if self.stop_event:
                self.stop_event.set()
        self.save_current_settings()
        self.Close()

    def set_table_chapter_status(self, chapter_index, status):
        self.table.SetItem(chapter_index, 3, status)

    def open_folder_with_explorer(self, folder_path):
        try:
            if platform.system() == 'Windows':
                subprocess.Popen(['explorer', folder_path])
            elif platform.system() == 'Linux':
                subprocess.Popen(['xdg-open', folder_path])
            elif platform.system() == 'Darwin':
                subprocess.Popen(['open', folder_path])
        except Exception as e:
            print(e)


class CoreThread(threading.Thread):
    def __init__(self, params, stop_event):
        super().__init__(daemon=True)
        self.params = params
        self.stop_event = stop_event

    def run(self):
        import audiblez.core as core
        core.main(**self.params, stop_event=self.stop_event, post_event=self.post_event)

    def post_event(self, event_name, **kwargs):
        EventObject, EVENT_CODE = EVENTS[event_name]
        event_object = EventObject()
        for k, v in kwargs.items():
            setattr(event_object, k, v)
        wx.PostEvent(wx.GetApp().GetTopWindow(), event_object)


def main():
    print('Starting GUI...')
    app = wx.App(False)
    frame = MainWindow(None, "Audiblez - Generate Audiobooks from E-books")
    frame.Show(True)
    frame.Layout()
    app.SetTopWindow(frame)
    print('Done.')
    app.MainLoop()


if __name__ == '__main__':
    main()
