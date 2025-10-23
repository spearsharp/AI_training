import 'dart:convert';
import 'package:http/http.dart' as http;

const String deepseekApiKey = "sk-9362d28569be445f9a9cb77056d28795";
const String deepseekApiUrl = "https://api.deepseek.com/v1/chat/completions";

class DeepseekService {
  Future<String> sendMessage(String message) async {
    final response = await http.post(
      Uri.parse(deepseekApiUrl),
      headers: {
        'Content-Type': 'application/json; charset=utf-8',
        'Accept-Charset': 'utf-8',
        'Authorization': 'Bearer $deepseekApiKey',
      },
      body: jsonEncode({
        "model": "deepseek-chat",
        "messages": [
          {"role": "user", "content": message},
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
      }),
    );

    if (response.statusCode == 200) {
      final data = jsonDecode(response.body);
      return data["choices"][0]["message"]["content"] ?? "No response.";
    } else {
      return "Error: ${response.statusCode}";
    }
  }
}
