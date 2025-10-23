import 'package:flutter/material.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:emoji_picker_flutter/emoji_picker_flutter.dart';
import 'deepseek_service.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      localizationsDelegates: const [
        GlobalMaterialLocalizations.delegate,
        GlobalWidgetsLocalizations.delegate,
        GlobalCupertinoLocalizations.delegate,
      ],
      supportedLocales: const [
        Locale('en'),
        Locale('zh'),
        Locale('zh', 'CN'),
        Locale('zh', 'TW'),
      ],
      theme: ThemeData(
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a purple toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const MyHomePage(title: 'Flutter AI Chatbot Deepseek'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});
  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  final DeepseekService _deepseekService = DeepseekService();
  final TextEditingController _controller = TextEditingController();
  final FocusNode _focusNode = FocusNode();
  final List<Map<String, String>> _messages = [];
  bool _isLoading = false;
  bool _isEmojiVisible = false;
  bool _isComposing = false; // true when IME (e.g., Pinyin) is composing

  @override
  void initState() {
    super.initState();
    _controller.addListener(_handleTextChanged);
  }

  void _handleTextChanged() {
    final composing = _controller.value.composing.isValid;
    if (composing != _isComposing) {
      setState(() => _isComposing = composing);
    }
  }

  void _toggleEmojiKeyboard() {
    if (_isEmojiVisible) {
      setState(() => _isEmojiVisible = false);
      _focusNode.requestFocus();
    } else {
      FocusScope.of(context).unfocus();
      setState(() => _isEmojiVisible = true);
    }
  }

  void _insertEmoji(String emoji) {
    final text = _controller.text;
    final selection = _controller.selection;
    final start = selection.start >= 0 ? selection.start : text.length;
    final end = selection.end >= 0 ? selection.end : text.length;
    final newText = text.replaceRange(start, end, emoji);
    _controller.text = newText;
    final cursorPos = start + emoji.length;
    _controller.selection = TextSelection.fromPosition(
      TextPosition(offset: cursorPos),
    );
  }

  void _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty) return;
    if (_controller.value.composing.isValid) {
      // Avoid sending while Pinyin/IME is composing; keep focus to commit.
      _focusNode.requestFocus();
      return;
    }
    setState(() {
      _messages.add({"role": "user", "content": text});
      _isLoading = true;
      _controller.clear();
      _isEmojiVisible = false;
    });
    final reply = await _deepseekService.sendMessage(text);
    setState(() {
      _messages.add({"role": "bot", "content": reply});
      _isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      resizeToAvoidBottomInset: true,
      appBar: AppBar(
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        title: Row(
          children: [
            SvgPicture.asset('assets/robot.svg', height: 28, width: 28),
            const SizedBox(width: 8),
            Text(widget.title),
          ],
        ),
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final msg = _messages[index];
                final isUser = msg["role"] == "user";
                final bubble = Container(
                  constraints: BoxConstraints(
                    maxWidth: MediaQuery.of(context).size.width * 0.7,
                  ),
                  decoration: BoxDecoration(
                    color: isUser ? Colors.blue[100] : Colors.grey[200],
                    borderRadius: BorderRadius.circular(12),
                  ),
                  padding: const EdgeInsets.symmetric(
                    vertical: 8,
                    horizontal: 12,
                  ),
                  child: Text(msg["content"] ?? ""),
                );

                final avatar = ClipOval(
                  child: SvgPicture.asset(
                    isUser ? 'assets/user.svg' : 'assets/robot.svg',
                    width: 32,
                    height: 32,
                    fit: BoxFit.cover,
                  ),
                );

                return Padding(
                  padding: const EdgeInsets.symmetric(
                    vertical: 6,
                    horizontal: 12,
                  ),
                  child: Row(
                    mainAxisAlignment: isUser
                        ? MainAxisAlignment.end
                        : MainAxisAlignment.start,
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: isUser
                        ? [
                            Flexible(child: bubble),
                            const SizedBox(width: 8),
                            avatar,
                          ]
                        : [
                            avatar,
                            const SizedBox(width: 8),
                            Flexible(child: bubble),
                          ],
                  ),
                );
              },
            ),
          ),
          if (_isLoading)
            const Padding(
              padding: EdgeInsets.all(8.0),
              child: CircularProgressIndicator(),
            ),
          AnimatedPadding(
            duration: const Duration(milliseconds: 200),
            curve: Curves.easeOut,
            padding: EdgeInsets.only(
              bottom: _isEmojiVisible
                  ? 0
                  : MediaQuery.of(context).viewInsets.bottom,
            ),
            child: SafeArea(
              top: false,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
                decoration: BoxDecoration(
                  color: Theme.of(context).scaffoldBackgroundColor,
                  border: const Border(
                    top: BorderSide(color: Color(0x14000000)),
                  ),
                ),
                child: Row(
                  children: [
                    IconButton(
                      tooltip: 'Emoji',
                      icon: Icon(
                        _isEmojiVisible
                            ? Icons.keyboard_alt_outlined
                            : Icons.emoji_emotions_outlined,
                      ),
                      onPressed: _isLoading ? null : _toggleEmojiKeyboard,
                    ),
                    Expanded(
                      child: Localizations.override(
                        context: context,
                        locale: const Locale('zh'),
                        child: TextField(
                          controller: _controller,
                          focusNode: _focusNode,
                          textInputAction: TextInputAction.send,
                          keyboardType: TextInputType.text,
                          textCapitalization: TextCapitalization.none,
                          enableSuggestions: true,
                          autocorrect: false,
                          smartDashesType: SmartDashesType.disabled,
                          smartQuotesType: SmartQuotesType.disabled,
                          enableIMEPersonalizedLearning: true,
                          decoration: const InputDecoration(
                            hintText: "Type your message...",
                            border: OutlineInputBorder(),
                            isDense: true,
                            contentPadding: EdgeInsets.symmetric(
                              horizontal: 12,
                              vertical: 10,
                            ),
                          ),
                          onTap: () {
                            if (_isEmojiVisible) {
                              setState(() => _isEmojiVisible = false);
                              // After hiding emoji panel, ensure keyboard shows.
                              _focusNode.requestFocus();
                            } else {
                              _focusNode.requestFocus();
                            }
                          },
                          onSubmitted: (_) {
                            if (!_isComposing) {
                              _sendMessage();
                            } else {
                              // If still composing, keep focus to commit text.
                              _focusNode.requestFocus();
                            }
                          },
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    IconButton(
                      icon: const Icon(Icons.send),
                      onPressed: (_isLoading || _isComposing)
                          ? null
                          : _sendMessage,
                    ),
                  ],
                ),
              ),
            ),
          ),
          // Emoji picker panel
          Offstage(
            offstage: !_isEmojiVisible,
            child: SizedBox(
              height: 280,
              child: EmojiPicker(
                onEmojiSelected: (category, emoji) {
                  _insertEmoji(emoji.emoji);
                },
                onBackspacePressed: () {
                  // Optional: Let the package handle backspace if textEditingController is set
                  final text = _controller.text;
                  final sel = _controller.selection;
                  if (sel.start > 0) {
                    final newText = text.replaceRange(
                      sel.start - 1,
                      sel.start,
                      '',
                    );
                    _controller.text = newText;
                    _controller.selection = TextSelection.fromPosition(
                      TextPosition(offset: sel.start - 1),
                    );
                  }
                },
                textEditingController: _controller,
                config: Config(
                  height: 280,
                  checkPlatformCompatibility: true,
                  searchViewConfig: SearchViewConfig(
                    backgroundColor: Colors.grey.withOpacity(0.1),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _controller.removeListener(_handleTextChanged);
    _controller.dispose();
    _focusNode.dispose();
    super.dispose();
  }
}
