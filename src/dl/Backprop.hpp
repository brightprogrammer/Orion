#include <memory>
#include <vector>

#include "../Tensor.hpp"

namespace Orion{
	typedef Tensor<double> ten; 

	class Function;

	class TensorVar{
		private:
			ten _value;
			bool _requires_grad;
			ten _grad;
			std::shared_ptr<Function> _func;
			std::vector<std::weak_ptr<TensorVar>> _consumers;
			bool _visited = false;
		public:
			TensorVar(const ten& t, bool require_grad = false) : _value(t), _requires_grad(require_grad), 
														_grad(t.dim())
			{
				reset_grad();
			}
			void reset_grad(){
				_grad.fill(0);
			}
			ten& value(){
				return _value;
			}
			ten& grad(){
				return _grad;
			}
			bool requires_grad() const{
				return _requires_grad;
			}
			void set_func(std::shared_ptr<Function> func){
				_func = func;
			}
			auto const& get_func() const{
				return _func;
			}
			void add_consumer(std::shared_ptr<TensorVar> const& v){
				_consumers.emplace_back(v);
			}
			auto const& get_consumers(){
				return _consumers;
			}
			bool is_visited() const{
				return _visited;
			}
			void set_visited(bool v){
				_visited = v;
			}
	};

	class Function{
		public:
			virtual std::shared_ptr<TensorVar> calc() = 0;
			virtual std::vector<std::shared_ptr<TensorVar>> const& get_inputs() const = 0;
			virtual void calc_grad() = 0;
	};

	class SumBP : public Function{
		public:
			SumBP(std::shared_ptr<TensorVar> const& x1, std::shared_ptr<TensorVar> const& x2){
				_in.emplace_back(x1);
				_in.emplace_back(x2);
			}
			std::shared_ptr<TensorVar> calc(){
				auto& _x1 = *_in[0];
				auto& _x2 = *_in[1];
				ten ym = _x1.value() + _x2.value();
				bool requires_grad = _x1.requires_grad() || _x2.requires_grad();
				auto out = std::make_shared<TensorVar>(ym, requires_grad);
				_out = out;
				return out;
			}

			void calc_grad(){
				auto& _x1 = *_in[0];
				auto& _x2 = *_in[1];
				ten t(_x1.value().dim());
				t.fill(1);	
				auto out = _out.lock();
				if(_x1.requires_grad())
					_x1.grad() += t%out->grad();
				if(_x2.requires_grad())
					_x2.grad() += t%out->grad();
			}

			std::vector<std::shared_ptr<TensorVar>> const& get_inputs() const{
				return _in;
			}
			std::weak_ptr<TensorVar> const& get_out() const{
				return _out;
			}
		private:
			std::vector<std::shared_ptr<TensorVar>> _in;
			std::weak_ptr<TensorVar> _out;
	};

	class Multiply : public Function{
		public:
			Multiply(std::shared_ptr<TensorVar> const& x1, std::shared_ptr<TensorVar> const& x2){
				_in.emplace_back(x1);
				_in.emplace_back(x2);
			}
			std::shared_ptr<TensorVar> calc(){
				auto& _x1 = *_in[0];
				auto& _x2 = *_in[1];
				ten& ym = _x1.value();
				ym %=_x2.value();
				bool requires_grad = _x1.requires_grad() || _x2.requires_grad();
				auto out = std::make_shared<TensorVar>(ym, requires_grad);
				_out = out;
				return out;
			}
			void calc_grad(){
				auto& _x1 = *_in[0];
				auto& _x2 = *_in[1];

				auto out = _out.lock();
				if(_x1.requires_grad())
					_x1.grad() += _x2.value()%out->grad();
				if(_x2.requires_grad())
					_x2.grad() += _x1.value()%out->grad();
			}
			std::vector<std::shared_ptr<TensorVar>> const& get_inputs() const{
				return _in;
			}
			auto const& get_out() const{
				return _out;
			}
		private:
			std::vector<std::shared_ptr<TensorVar>> _in;
			std::weak_ptr<TensorVar> _out;
	};

	class Subtract : public Function{
		public:
			Subtract(std::shared_ptr<TensorVar> const& x1, std::shared_ptr<TensorVar> const& x2){
				_in.emplace_back(x1);
				_in.emplace_back(x2);
			}
			std::shared_ptr<TensorVar> calc(){
				auto& _x1 = *_in[0];
				auto& _x2 = *_in[1];
				ten ym = _x1.value() - _x2.value();
				bool requires_grad = _x1.requires_grad() || _x2.requires_grad();
				auto out = std::make_shared<TensorVar>(ym, requires_grad);
				_out = out;
				// _out->_func = make_shared<SumBP>(*this);
				return out;
			}
			void calc_grad(){
				auto& _x1 = *_in[0];
				auto& _x2 = *_in[1];
				ten t{_x1.value().dim()};
				t.fill(1);

				auto out = _out.lock();
				if(_x1.requires_grad())
					_x1.grad() += t%out->grad();
				if(_x2.requires_grad())			
					_x2.grad() -= t%out->grad();
			}
			std::vector<std::shared_ptr<TensorVar>> const& get_inputs() const{
				return _in;
			}
			auto const& get_out() const{
				return _out;
			}
		private:
			std::vector<std::shared_ptr<TensorVar>> _in;
			std::weak_ptr<TensorVar> _out;
	};
	class Power : public Function{
		public:
			Power(std::shared_ptr<TensorVar> const& x1, int n) : _n(n){
				_in.emplace_back(x1);
			}
			std::shared_ptr<TensorVar> calc(){
				auto& _x1 = *_in[0];
				// TODO: Implement Pow in Tenso

				ten ym = Orion::pow(_x1.value(), _n);
				auto out = std::make_shared<TensorVar>(ym, _x1.requires_grad());
				_out = out;
				return out;
			}
			void calc_grad(){
				auto& _x1 = *_in[0];
				ten g = Orion::pow(_x1.value(), _n - 1);
				g %= _n;
				auto out = _out.lock();
				if(_x1.requires_grad())
					_x1.grad() += g%out->grad();
			}
			std::vector<std::shared_ptr<TensorVar>> const& get_inputs() const{
				return _in;
			}
			std::weak_ptr<TensorVar> const& get_out() const{
				return _out;
			}
		private:
			std::vector<std::shared_ptr<TensorVar>> _in;
			int _n;
			std::weak_ptr<TensorVar> _out;
	};
	class Exp : public Function{
		public:
			Exp(std::shared_ptr<TensorVar> const& x1){
				_in.emplace_back(x1);
			}
			std::shared_ptr<TensorVar> calc(){
				auto& _x1 = *_in[0];
				ten ym = exp_t(_x1.value());
				auto out = std::make_shared<TensorVar>(ym, _x1.requires_grad());
				_out = out;
				return out;
			}	
			void calc_grad(){
				auto& _x1 = *_in[0];
				auto out = _out.lock();
				if(_x1.requires_grad())
					_x1.grad() += out->value()%out->grad();
			}
			std::vector<std::shared_ptr<TensorVar>> const& get_inputs() const{
				return _in;
			}
			std::weak_ptr<TensorVar> const& get_out() const{
				return _out;
			}
		private:
			std::vector<std::shared_ptr<TensorVar>> _in;
			std::weak_ptr<TensorVar> _out;
	};

	class MatMul : public Function{
		public:
			MatMul(std::shared_ptr<TensorVar> const& x1, std::shared_ptr<TensorVar> const& x2){
				_in.emplace_back(x1);
				_in.emplace_back(x2);
			}
			std::shared_ptr<TensorVar> calc(){
				auto& _x1 = *_in[0];
				auto& _x2 = *_in[1];
				ten ym = _x1.value() * _x2.value();
				auto out = std::make_shared<TensorVar>(ym, _x1.requires_grad() || _x2.requires_grad());
				_out = out;
				return out;
			}
			void calc_grad(){
				auto& _x1 = *_in[0];
				auto& _x2 = *_in[1];
				auto out = _out.lock();
				ten const& grad = out->grad();

				if(_x1.requires_grad())
					_x1.grad() += grad * _x2.value().t();
				if(_x2.requires_grad())
					_x2.grad() += _x1.value().t() * grad;
			}
			std::vector<std::shared_ptr<TensorVar>> const& get_inputs() const{
				return _in;
			}
			std::weak_ptr<TensorVar> const& get_out() const{
				return _out;
			}
		private:
			std::vector<std::shared_ptr<TensorVar>> _in;
			std::weak_ptr<TensorVar> _out;
	};

	class Where : public Function{
		public:
			Where(ten predicate, std::shared_ptr<TensorVar> const& x1, std::shared_ptr<TensorVar> const& x2) : _predicate(std::move(predicate)){
				_in.emplace_back(x1);
				_in.emplace_back(x2);
			}
			std::shared_ptr<TensorVar> calc(){
				auto& _x1 = *_in[0];
				auto& _x2 = *_in[1];

				ten ym = _predicate%_x1.value() + (1-_predicate)%_x2.value();
				auto out = std::make_shared<TensorVar>(ym, _x1.requires_grad() || _x2.requires_grad());
				_out = out;
				return out;
			}
			void calc_grad(){
				auto& _x1 = *_in[0];
				auto& _x2 = *_in[1];
				auto out = _out.lock();

				if(_x1.requires_grad())
					_x1.grad() += _predicate%out->grad();
				if(_x2.requires_grad())
					_x2.grad() += (1 - _predicate)%out->grad();
			}
			std::vector<std::shared_ptr<TensorVar>> const& get_inputs() const{
				return _in;
			}
			std::weak_ptr<TensorVar> const& get_out() const{
				return _out;
			}
		private:
			std::vector<std::shared_ptr<TensorVar>> _in;
			std::weak_ptr<TensorVar> _out;
			ten _predicate;
	};

	// class Mean : public Function{
	// 	public:
	// 		Mean(std::shared_ptr<TensorVar> const& x1){
	// 			_in.emplace_back(x1);
	// 		}
	// 		std::shared_ptr<TensorVar> calc(){
	// 			auto& _x1 = *_in[0];
	// 			mat ym = mean(_x1.value(), 1);
	// 			auto out = std::make_shared<TensorVar>(ym, _x1.requires_grad());
	// 			_out = out;
	// 			return out;
	// 		}
	// 		void calc_grad(){
	// 			auto& _x1 = *_in[0];
	// 			ten 
	// 			ym /= _x1.value().n_cols;
	// 			auto out = _out.lock();			
	// 			_x1.grad() += ym*as_scalar(out->grad());
	// 		}
	// 		std::vector<std::shared_ptr<TensorVar>> const& get_inputs() const{
	// 			return _in;
	// 		}
	// 		std::weak_ptr<TensorVar> const& get_out() const{
	// 			return _out;
	// 		}
	// 	private:
	// 		std::vector<std::shared_ptr<TensorVar>> _in;
	// 		std::weak_ptr<TensorVar> _out;
	// };

	inline auto operator*(std::shared_ptr<TensorVar> const& x, std::shared_ptr<TensorVar> const& y){
		auto m = std::make_shared<MatMul>(x, y);
		auto z = m->calc();
		z->set_func(m);
		return z;
	}

	inline auto operator+(std::shared_ptr<TensorVar> const& x, std::shared_ptr<TensorVar> const& y){
		auto s = std::make_shared<SumBP>(x, y);
		auto z = s->calc();
		z->set_func(s);
		return z;
	}

	inline auto operator+(std::shared_ptr<TensorVar> const& x, double scalar){
		auto const& m = x->value();
		ten sm(m.dim());
		sm.fill(scalar);

		auto sv = std::make_shared<TensorVar>(sm, false);
		return sv + x;
	}
	inline auto operator+(double scalar, std::shared_ptr<TensorVar> const& x){
		return x + scalar;
	}

	inline auto operator-(std::shared_ptr<TensorVar> const& x, std::shared_ptr<TensorVar> const& y){
		auto s = std::make_shared<Subtract>(x, y);
		auto z = s->calc();
		z->set_func(s);
		return z;
	}
	inline auto operator-(std::shared_ptr<TensorVar> const& x, double scalar){
		auto const& m = x->value();
		ten sm(m.dim());
		sm.fill(scalar);
		auto sv = std::make_shared<TensorVar>(sm, false);

		return x - sv;
	}
	inline auto operator-(double scalar, std::shared_ptr<TensorVar> const& x){
		auto const& m = x->value();
		ten sm(m.dim());
		sm.fill(scalar);
		auto sv = std::make_shared<TensorVar>(sm, false);

		return sv - x;
	}

	inline auto operator-(std::shared_ptr<TensorVar> const& x){
		auto& xm = x->value();
		ten ym(xm.dim());
		ym.fill(0);
		auto y = std::make_shared<TensorVar>(ym, false);
		return (y - x);
	}

	inline auto operator%(std::shared_ptr<TensorVar> const& x, std::shared_ptr<TensorVar> const& y){
		auto mut = std::make_shared<Multiply>(x, y);
		auto z = mut->calc();
		z->set_func(mut);
		return z;
	}

	inline auto exp(std::shared_ptr<TensorVar> const& x){
		auto e = std::make_shared<Exp>(x);
		auto z = e->calc();
		z->set_func(e);

		return z;
	}
	inline auto pow(std::shared_ptr<TensorVar> const& x, int pow){
		auto p = std::make_shared<Power>(x, pow);
		auto z = p->calc();
		z->set_func(p);

		return z;
	}

	inline auto where(const ten &predicate, std::shared_ptr<TensorVar> const& x, std::shared_ptr<TensorVar> const& y){
		auto w = std::make_shared<Where>(predicate, x, y);
		auto z = w->calc();
		z->set_func(w);

		return z;
	}

	void display(std::shared_ptr<TensorVar> const& x){
		std::cout << x->value().rank() << "->";
		auto const& func = x->get_func();
		if(func != nullptr){
			// func->repr();
			auto& inputs = func->get_inputs();
			// for(auto const& i : inputs){
			// 	cout << arma::size(i->m()) << ' ';
			// }
			// cout << endl;
			// cout << arma::size(inputs[0]->m()) << endl;
			std::cout << inputs.size() << std::endl;
			for(auto const& i : inputs){
				display(i);
			}
		}
		std::cout << std::endl;
	}


	// inline auto mean(std::shared_ptr<TensorVar> const& x){
	// 	auto m = std::make_shared<Mean>(x);
	// 	auto z = m->calc();
	// 	z->set_func(m);
	// 	return z;
	// }

	void find_consumers(std::shared_ptr<TensorVar> const& x){
		auto& func = x->get_func();
		if(func != nullptr){
			auto const& inputs = func->get_inputs();
			for(auto const& i : inputs){
				i->add_consumer(x);
				find_consumers(i);
			}
		}
	}


	void build_grad(std::shared_ptr<TensorVar> const& x){
		if(!x->is_visited()){
			auto const& consumers = x->get_consumers();
			for(auto const& i : consumers){
				build_grad(i.lock());
			}
			x->set_visited(1);
			auto const& func = x->get_func();
			if(func != nullptr)
				func->calc_grad();
		}
	}
	void backward(std::shared_ptr<TensorVar> const& z, std::vector<std::shared_ptr<TensorVar>> vx){
		find_consumers(z);

		auto& grad = z->grad();
		grad.fill(1);
		// z->set_visited();
		for(auto const& x : vx){
			build_grad(x);
		}
	}

}